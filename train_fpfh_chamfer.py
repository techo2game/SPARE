"""
Train learnable fusion network for FPFH + Geometry (mean+normal) anomaly detection.

This script trains a neural network to learn optimal fusion of:
- FPFH branch: Surface descriptors
- Geometry branch: Patch mean position + surface normal

Config files are saved to logs/<category>/<timestamp>/ for reproducibility.
"""
import os
import sys
import argparse
import numpy as np
from compare_geometry_fpfh import apply_transformation_torch
from registration.methods.parenet_torch import parenet_torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import pickle
import time
import shutil
import yaml
from easydict import EasyDict
from sklearn.metrics import roc_auc_score
from torch_cluster import knn
from torch_scatter import scatter_max
# Add cuda_ops to path for GPU FPFH
cuda_ops_dir = os.path.join(os.path.dirname(__file__), 'cuda_ops')
if cuda_ops_dir not in sys.path:
    sys.path.insert(0, cuda_ops_dir)
# Utilities
from utils.misc import seed_all, get_logger, get_new_log_dir
from utils.config import cfg_from_yaml_file
from utils.data_loading import load_reference_data_spatial_chamfer, find_test_samples, load_test_pointcloud
from utils.artifacts import dataset_dir_name
from utils.metrics import compute_metrics
from utils.dataset import all_real3dad_cates


# FPFH
# Loss functions
from utils.loss import  patch_contrastive_loss, get_loss_function
# GPU acceleration
from torch_cluster import fps
from pykeops.torch import LazyTensor
# Models
from models.fusion_models import *
# Geometry utilities
from utils.geometry import compute_geometry_scores_learned_fusion
# Feature computation
from utils.patch_features import compute_patch_features_fpfh_chamfer_batch
# Sampling utilities
from utils.sampling import extract_patches_gpu_fps, collate_fn_with_patches
# FPFH utilities
from utils.fpfh_utils import compute_fpfh_features
import fpsample

from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
def voxel_downsample_indices(points, target_points):
    """
    Voxel downsampling that returns indices to original points.
    Adaptively finds voxel size to achieve ~target point count.

    Args:
        points: (N, 3) torch.Tensor on GPU
        target_points: Target number of points

    Returns:
        indices: (M,) indices of selected points
    """
    # Estimate initial voxel size from bounding box
    extent = points.max(dim=0).values - points.min(dim=0).values
    volume = extent.prod().item()
    voxel_size = (volume / target_points) ** (1/3)

    # Binary search for good voxel size
    low, high = voxel_size * 0.1, voxel_size * 5.0

    for _ in range(10):
        mid = (low + high) / 2
        cluster = voxel_grid(points, size=mid)
        _, indices = consecutive_cluster(cluster)
        n_unique = indices.shape[0]

        if n_unique < target_points * 0.9:
            high = mid
        elif n_unique > target_points * 1.1:
            low = mid
        else:
            break

    return indices


def precompute_test_features(test_samples, ref_data, config, category, device, args, force_recompute=False, registration_method='parenet'):
    """
    Precompute and cache test features to avoid recomputation during evaluation.

    Args:
        test_samples: List of (subdir, test_file, test_file_path) tuples
        ref_data: Reference data with spatial_bank
        config: Config object
        category: Category name
        device: torch device
        args: Arguments with num_patches, patch_size, k_centers
        force_recompute: If True, ignore existing cache
        registration_method: 'default' or 'patch_dropout' for visualization

    Returns:
        List of dicts containing precomputed features for each test sample
    """
    from utils.data_loading import load_test_pointcloud

    # Setup cache path
    dataset_dir = dataset_dir_name(config.dataset)

    cache_dir = os.path.join('precomputed_features', dataset_dir, category, 'test')
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = f"patches{args.num_patches2}_size{args.patch_size}_k{args.k_centers}"
    cache_path = os.path.join(cache_dir, f'{cache_key}.pkl')

    # Try to load from cache
    if not force_recompute and os.path.exists(cache_path):
        #print(f"Loading precomputed test features from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"✓ Loaded {len(cached_data)} test samples from cache")
            return cached_data
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), recomputing...")

    # Cache miss - compute features
    print(f"Precomputing test features for {len(test_samples)} samples...")

    # Move reference data to GPU
    ref_points_gpu = torch.from_numpy(ref_data['points']).float().to(device)
    # reference_patch_centers = ref_data['spatial_bank']['centers'].to(device)
    # reference_patch_fpfh = ref_data['spatial_bank']['fpfh_matrix'].to(device)
    # reference_means = ref_data['spatial_bank']['means_matrix'].to(device)
    # reference_normals = ref_data['spatial_bank']['normals_matrix'].to(device)

    cached_test_data = []

    for subdir, test_file, test_file_path in tqdm(test_samples, desc="Precomputing test features"):
        # Load test point cloud (same as TestPointCloudDataset)
        test_data = load_test_pointcloud(test_file_path, config.dataset, subdir, is_evaluate=True)
        points_normalized = test_data['points_normalized']
        gt_mask = test_data['gt_mask']

        pixel_indices = test_data.get('pixel_indices')
        original_shape = test_data.get('original_shape')

        # Determine label
        if config.dataset in ['Real3DAD', 'ShapeNetAD']:
            is_normal = 'good' in test_file.lower() or 'positive' in test_file.lower()
            label = 0 if is_normal else 1
        else:
            label = 0 if subdir == 'good' else 1

        # Handle missing GT mask
        if gt_mask is None:
            gt_mask = np.zeros(len(points_normalized), dtype=np.float32) if label == 0 else np.ones(len(points_normalized), dtype=np.float32)

        # Convert to GPU
        test_full = torch.from_numpy(points_normalized).float().to(device)
        gt_mask_full = gt_mask  # Keep full gt_mask for storage

        # Subsample using voxel downsampling if too large (same as test_fusion_model)
        max_points = 150000
        if test_full.shape[0] > max_points:
            sample_indices = voxel_downsample_indices(test_full, max_points)
            test = test_full[sample_indices]
            sample_indices = sample_indices.cpu().numpy()
        else:
            test = test_full
            sample_indices = None

        # Setup registration cache path
        reg_cache_dir = os.path.join('reg_output', dataset_dir, category, 'transforms')
        os.makedirs(reg_cache_dir, exist_ok=True)
        reg_cache_path = os.path.join(reg_cache_dir, f"{subdir}_{test_file}.npy")


        transform, test_aligned = register_pointcloud_cached(
            test, ref_points_gpu, reg_cache_path,
            use_cache=not force_recompute,
            registration_method=registration_method
        )

        diameter = torch.norm(test_aligned.max(dim=0).values - test_aligned.min(dim=0).values)
        voxel_size = diameter * 0.05
        test_fpfh = compute_fpfh_features(test_aligned, voxel_size.item())



        patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, patch_indices = extract_patches_gpu_fps(
            test_aligned, test_fpfh, args.num_patches2, args.patch_size
        )

        # Store raw data on CPU
        cached_sample = {
            'subdir': subdir,
            'test_file': test_file,
            'label': label,
            'patch_points': patch_points_gpu.cpu(),   # Raw patch points
            'patch_fpfh': patch_fpfh_gpu.cpu(),       # Raw patch FPFH
            'patch_centers': patch_centers_gpu.cpu(), # Patch centers
            'patch_indices': patch_indices,           # List of numpy arrays
            'points_normalized': points_normalized,   # For remapping
            'gt_mask': gt_mask_full,                  # Full gt_mask (matches points_normalized)
            'sample_indices': sample_indices,
            'pixel_indices': pixel_indices,
            'original_shape': original_shape,
            'test_aligned_shape': test_aligned.shape[0],  # For remapping
        }

        cached_test_data.append(cached_sample)

    # Save to cache
    print(f"Saving precomputed test features to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_test_data, f)

    print(f"✓ Precomputed and cached {len(cached_test_data)} test samples")

    return cached_test_data

class TestPrecomputedDataset(Dataset):
    """
    Dataset for precomputed test features.
    Simply loads cached features from memory.
    """
    def __init__(self, cached_test_data):
        self.data = cached_test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def test_collate_fn(batch):
    """
    Collate function for batched test evaluation.
    Simply returns a list of samples (no stacking since we'll concatenate patches).
    """
    return batch

def remap_sampled_to_full(point_scores_sampled, sample_indices, num_points_full, point_cloud_full):
    """
    Remap scores from sampled points to full point cloud using index-based mapping.

    Unsampled points get the score of their nearest neighbor in the sampled set.

    Args:
        point_scores_sampled: (num_sampled,) scores for sampled points
        sample_indices: (num_sampled,) indices of sampled points in full cloud, or None if no sampling
        num_points_full: total number of points in full cloud
        point_cloud_full: (num_points_full, 3) full point cloud coordinates (needed for NN lookup)

    Returns:
        point_scores_full: (num_points_full,) scores for all points
    """
    if sample_indices is None:
        # No sampling was performed
        return point_scores_sampled

    # Assign scores to sampled points
    point_scores_full = np.zeros(num_points_full, dtype=np.float32)
    point_scores_full[sample_indices] = point_scores_sampled

    # For unsampled points, assign NN score from sampled points
    unsampled_mask = np.ones(num_points_full, dtype=bool)
    unsampled_mask[sample_indices] = False
    unsampled_indices = np.where(unsampled_mask)[0]

    if len(unsampled_indices) > 0:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(point_cloud_full[sample_indices])
        _, nn_indices = nn.kneighbors(point_cloud_full[unsampled_indices])
        point_scores_full[unsampled_indices] = point_scores_sampled[nn_indices.flatten()]

    return point_scores_full

def remap_patch_scores_to_points(patch_scores, patch_indices, num_points, point_cloud=None):
    """
    Remap patch scores to point scores using index-based mapping (PyTorch version).

    Each patch score is assigned to all points in its KNN neighborhood.
    Points in multiple patches get averaged scores.
    Points not in any patch get NN score (rare with overlapping patches).

    Args:
        patch_scores: (num_patches,) numpy array of patch scores
        patch_indices: list of (patch_size,) arrays - KNN indices for each patch
        num_points: total number of points
        point_cloud: (num_points, 3) torch tensor on GPU - only needed if uncovered points exist

    Returns:
        point_scores: (num_points,) numpy array
    """
    point_scores = np.zeros(num_points, dtype=np.float32)
    point_counts = np.zeros(num_points, dtype=np.int32)

    # Assign each patch score to all its KNN points
    for i, indices in enumerate(patch_indices):
        point_scores[indices] += patch_scores[i]
        point_counts[indices] += 1

    # Average scores for points in multiple patches
    mask = point_counts > 0
    point_scores[mask] = point_scores[mask] / point_counts[mask]

    # For points not in any patch, assign NN score (rare with overlapping patches)
    uncovered_mask = point_counts == 0
    if uncovered_mask.any() and point_cloud is not None:
        from pykeops.torch import LazyTensor

        covered_indices = np.where(mask)[0]
        uncovered_indices = np.where(uncovered_mask)[0]

        # Get covered and uncovered points on GPU
        device = point_cloud.device
        covered_points = point_cloud[covered_indices]      # (N_covered, 3)
        uncovered_points = point_cloud[uncovered_indices]  # (N_uncovered, 3)

        # PyKeOps KNN - memory efficient (no materialized distance matrix)
        X_i = LazyTensor(uncovered_points[:, None, :])  # (N_uncovered, 1, 3)
        X_j = LazyTensor(covered_points[None, :, :])    # (1, N_covered, 3)
        D_ij = ((X_i - X_j) ** 2).sum(-1)               # Symbolic distance matrix
        nn_indices = D_ij.argKmin(1, dim=1).squeeze()   # (N_uncovered,) - memory efficient!

        # Map scores
        point_scores[uncovered_indices] = point_scores[covered_indices[nn_indices.cpu().numpy()]]

    return point_scores

def register_pointcloud_cached(test_points, reference_points, cache_path, use_cache=True, registration_method='default'):
    """
    Register test pointcloud to reference with caching support.

    Checks if a cached transform exists at cache_path. If yes, loads and applies it.
    Otherwise, performs registration and saves the transform to cache.

    Args:
        test_points: Test point cloud torch tensor (N, 3) on GPU
        reference_points: Reference point cloud torch tensor (M, 3) on GPU
        cache_path: Path to cache file (.pkl)
        use_cache: If False, always perform registration (for timing measurements)
        registration_method: 'default' (9dof) or 'patch_dropout' (more robust)

    Returns:
        tuple: (transform_torch, test_aligned_torch) - both on GPU
    """
    device = test_points.device

    if use_cache and os.path.exists(cache_path):
        # Load cached transform
        with open(cache_path, 'rb') as f:
            transform = pickle.load(f)
        transform = transform.to(device)
        test_aligned = apply_transformation_torch(test_points, transform)
    else:
        transform = parenet_torch(
            test_points, reference_points
        )

        test_aligned = apply_transformation_torch(test_points, transform)

        # Save transform to cache (only if use_cache is True)
        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(transform, f)

    return transform, test_aligned


def evaluate_on_test_set_with_model(model, ref_data, config, category, device, args, visualize=False, use_cache=True, batch_size=8):
    """
    Evaluate trained fusion model on test set using BATCHED processing (MUCH FASTER).

    Args:
        model: Trained fusion model
        ref_data: Reference data with spatial_bank
        config: Config object
        category: Category name
        device: torch device
        args: Arguments with num_patches, patch_size, k_centers
        visualize: If True, visualize each test sample with PyVista
        use_cache: If True, use precomputed test features (much faster)
        batch_size: Number of test samples to process in parallel (default: 8)
    """


    if visualize:
        try:
            from registration.pyvista_visualizer import visualize_anomaly_scores
        except ImportError:
            print("Warning: pyvista_visualizer not available, disabling visualization")
            visualize = False

    dataset_path = config.dataset_path
    dataset_type = config.dataset

    # Map dataset type to directory name
    dataset_dir = dataset_dir_name(dataset_type)

    # Find test samples
    test_samples = find_test_samples(dataset_path, category, dataset_type, test_file=None)

    if use_cache:

        cached_test_data = precompute_test_features(
            test_samples, ref_data, config, category, device, args, force_recompute=args.recompute, registration_method="parenet"
        )
        test_dataset = TestPrecomputedDataset(cached_test_data)
    else:
        raise NotImplementedError("Evaluation only supports precomputed test features (use_cache=True)")

    # Create DataLoader with batching
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Already precomputed, no need for workers
        collate_fn=test_collate_fn
    )

    # Move reference data to device once
    reference_patch_centers = ref_data['spatial_bank']['centers'].to(device)
    reference_patch_fpfh = ref_data['spatial_bank']['fpfh_matrix'].to(device)
    reference_means = ref_data['spatial_bank']['means_matrix'].to(device)
    reference_normals = ref_data['spatial_bank']['normals_matrix'].to(device)

    # Evaluation containers
    image_preds, image_labels = [], []
    pixel_preds, pixel_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating test set")):

            batch_size_actual = len(batch)

            # Prepare batch data
            all_patch_points = []
            all_patch_fpfh = []
            all_patch_centers = []
            patch_counts = [] 

            # Metadata for remapping
            metadata = []

            for sample in batch:
                if use_cache:
                    # Load cached data
                    patch_points = sample['patch_points'].to(device)
                    patch_fpfh = sample['patch_fpfh'].to(device)
                    patch_centers = sample['patch_centers'].to(device)

                    metadata.append({
                        'patch_indices': sample['patch_indices'],
                        'points_normalized': sample['points_normalized'],
                        'gt_mask': sample['gt_mask'],
                        'label': sample['label'],
                        'subdir': sample['subdir'],
                        'test_file': sample['test_file'],
                        'sample_indices': sample['sample_indices'],
                        'test_aligned_shape': sample['test_aligned_shape']
                    })
                else:
                    raise NotImplementedError("Batched evaluation only supports use_cache=True for now")

                # Accumulate patches
                all_patch_points.append(patch_points)
                all_patch_fpfh.append(patch_fpfh)
                all_patch_centers.append(patch_centers)
                patch_counts.append(len(patch_points))

            # Concatenate all patches from batch: (B*P, ...)
            all_patch_points = torch.cat(all_patch_points, dim=0)      # (B*P, patch_size, 3)
            all_patch_fpfh = torch.cat(all_patch_fpfh, dim=0)          # (B*P, 33)
            all_patch_centers = torch.cat(all_patch_centers, dim=0)    # (B*P, 3)

            # Single forward pass for entire batch!
            all_patch_scores = model(
                all_patch_points,          # (B*P, patch_size, 3)
                all_patch_fpfh,            # (B*P, 33)
                all_patch_centers,         # (B*P, 3)
                reference_patch_centers,   # (R, 3)
                reference_means,           # (R, N, 3)
                reference_normals,         # (R, N, 3)
                reference_patch_fpfh,      # (R, N, 33)
                k_centers=args.k_centers
            )

            # Split scores back to individual samples
            patch_scores_split = torch.split(all_patch_scores, patch_counts)

            # Process each sample in batch
            for i, (patch_scores_gpu, meta) in enumerate(zip(patch_scores_split, metadata)):
                patch_scores_cpu = patch_scores_gpu.cpu().numpy()

                # Get metadata
                patch_indices = meta['patch_indices']
                points_normalized = meta['points_normalized']
                gt_mask = meta['gt_mask']
                label = meta['label']
                subdir = meta['subdir']
                test_file = meta['test_file']
                sample_indices = meta['sample_indices']
                test_aligned_shape = meta['test_aligned_shape']

                # Remap to points 
                if isinstance(patch_indices, torch.Tensor):
                    patch_indices = [patch_indices[i].cpu().numpy() for i in range(len(patch_indices))]

                point_scores_sampled = remap_patch_scores_to_points(
                    patch_scores_cpu, patch_indices, test_aligned_shape, None
                )

                # Remap from sampled to full
                point_scores = remap_sampled_to_full(
                    point_scores_sampled, sample_indices, len(points_normalized), points_normalized
                )

                if visualize:
                    visualize_anomaly_scores(
                        points=points_normalized,
                        scores=point_scores,
                        gt_mask=gt_mask,
                        title=f"{subdir}/{test_file} (Label: {'Anomaly' if label == 1 else 'Normal'})",
                        show=True
                    )

                # Collect predictions
                image_preds.append(np.max(point_scores))
                image_labels.append(label)
                pixel_preds.extend(point_scores)
                pixel_labels.extend(gt_mask)

                del patch_scores_cpu, point_scores_sampled, point_scores

            del all_patch_points, all_patch_fpfh, all_patch_centers, all_patch_scores, patch_scores_split

            if device.type == 'cuda' and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    # Compute metrics
    if len(image_labels) == 0:
        return {'i_auroc': 0.0, 'p_auroc': 0.0, 'num_samples': 0}

    image_preds = np.array(image_preds)
    image_labels = np.array(image_labels)

    i_auroc = roc_auc_score(image_labels, image_preds) if len(np.unique(image_labels)) >= 2 else 0.0
    p_auroc = roc_auc_score(pixel_labels, pixel_preds) if len(np.unique(pixel_labels)) >= 2 else 0.0

    return {
        'i_auroc': i_auroc,
        'p_auroc': p_auroc,
        'num_samples': len(image_labels)
    }

# ============================================================================
# Dataset Class
# ============================================================================
class SyntheticAnomalyFPFHGeometryDataset(Dataset):
    """
    Dataset for image-level contrastive learning.

    Returns pairs: (anomaly sample, random normal sample)
    """
    def __init__(self, synthetic_dir, category, args, config):
        """
        Args:
            synthetic_dir: Path to synthetic_data/{category}/
            category: Category name
            args: Arguments
            config: Config for loading normal training samples
        """
        self.synthetic_dir = synthetic_dir
        self.category = category
        self.args = args
        self.config = config

        # Setup cache
        dataset_dir = dataset_dir_name(config.dataset)

        cache_dir = os.path.join('precomputed_features', dataset_dir, category)
        os.makedirs(cache_dir, exist_ok=True)

        cache_key = f"patches{args.num_patches}_size{args.patch_size}"
        cache_path = os.path.join(cache_dir, f'{cache_key}.pkl')


        if os.path.exists(cache_path):
            print(f"Loading precomputed features from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.data_cache = cached_data['anomaly_samples']
                print(f"✓ Loaded {len(self.data_cache)} anomaly samples from cache")
                return  # Skip computation
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}), recomputing...")

        print("Cache not found, precomputing features...")

        synthetic_path = os.path.join(synthetic_dir, 'train_synthetic')
        if not os.path.exists(synthetic_path):
            synthetic_path = synthetic_dir  # Fallback

        all_files = sorted([f for f in os.listdir(synthetic_path) if f.endswith('.pkl')])
        if args.max_synthetic_samples is not None:
            all_files = all_files[:args.max_synthetic_samples]
        if len(all_files) == 0:
            raise FileNotFoundError(f"No .pkl files found in {synthetic_path}")

        print(f"  Loading {len(all_files)} anomaly samples...")

        # PRELOAD ANOMALY DATA INTO RAM
        temp_data = []
        sizes = []
        for filename in tqdm(all_files, desc="Loading anomaly data"):
            filepath = os.path.join(synthetic_path, filename)
            data = torch.load(filepath, weights_only=True)
            temp_data.append(data)
            sizes.append(len(data['points']))

        # Use minimum size - downsample all to smallest
        fixed_size = min(sizes)
        print(f"  Downsampling all samples to: {fixed_size} points")

        # Load reference data for feature computation
        ref_data = load_reference_data_spatial_chamfer(category, dataset=dataset_dir)

        # Move reference data to GPU temporarily
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        reference_patch_centers = ref_data['spatial_bank']['centers'].to(device)
        reference_patch_fpfh = ref_data['spatial_bank']['fpfh_matrix'].to(device)
        reference_means = ref_data['spatial_bank']['means_matrix'].to(device)
        reference_normals = ref_data['spatial_bank']['normals_matrix'].to(device)

        print(f"  Precomputing FPFH for anomaly samples...")
        self.data_cache = []
        for data in tqdm(temp_data, desc="Computing FPFH"):
            points = data["points"]  # (N, 3) tensor
            mask = data["mask"]  # (N,) tensor
            N = points.size(0)

            if N > fixed_size:
                # Random downsampling
                idx = torch.randperm(N, device=points.device)[:fixed_size]
                points = points[idx]
                mask = mask[idx]

            # Move to GPU and compute FPFH
            points_gpu = points.to(device)
            diameter = torch.norm(points_gpu.max(dim=0).values - points_gpu.min(dim=0).values)
            voxel_size = diameter * 0.05
            fpfh_gpu = compute_fpfh_features(points_gpu, voxel_size.item())
            # Store raw data on CPU (NO patch computation here)
            data["points"] = points_gpu.cpu()  # Store normalized points
            data["fpfh"] = fpfh_gpu.cpu()  # Store FPFH
            data["mask"] = mask.cpu()  # Store mask
            data["split"] = data.get('split', 'train')
            self.data_cache.append(data)

        print(f"  ✓ Precomputed {len(self.data_cache)} anomaly samples")

        # Save to cache (no normal samples needed)
        print(f"  Saving to cache: {cache_path}")
        cache_data = {
            'anomaly_samples': self.data_cache,
            'num_patches': args.num_patches,
            'patch_size': args.patch_size,
            'category': category,
            'dataset': config.dataset
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  ✓ Cache saved!")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        """Return raw point cloud data (NO precomputed patches)."""
        # Return raw data - patches will be extracted dynamically in training loop
        anomaly_data = self.data_cache[idx]

        return {
            # Raw data (patches extracted dynamically)
            'points': anomaly_data['points'],  # (N, 3)
            'fpfh': anomaly_data['fpfh'],      # (N, 33)
            'mask': anomaly_data['mask'],      # (N,)
            # Metadata
            'split': anomaly_data.get('split', 'train')
        }

# ============================================================================
# Training Function
# ============================================================================
def train_fusion_network(args):
    """Main training loop for FPFH + Geometry fusion network."""

    # Load dataset config
    config = args.dataset_config
    category = args.category
    seed_all(args.seed)

    # Create log directory
    log_root = os.path.join('logs', category)
    log_dir = get_new_log_dir(root=log_root)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize logger
    logger = get_logger('train_fusion', log_dir)
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Category: {category}")

    # Save configs for reproducibility
    shutil.copy(args.training_config_path, os.path.join(log_dir, 'training_config.yaml'))
    shutil.copy(args.dataset_config_path, os.path.join(log_dir, 'dataset_config.yaml'))
    # The YAML copies above miss CLI overrides (--epochs, --batch_size, ...):
    # also record the settings this run actually used.
    effective = {k: v for k, v in args.items()
                 if isinstance(v, (bool, int, float, str, list, type(None)))}
    with open(os.path.join(log_dir, 'effective_config.yaml'), 'w') as f:
        yaml.safe_dump(effective, f, sort_keys=True)
    logger.info(f"Saved training config: training_config.yaml")
    logger.info(f"Saved dataset config: dataset_config.yaml")
    logger.info(f"Saved effective settings: effective_config.yaml")


    dataset_name = config.dataset
    synthetic_dir = os.path.join(args.synthetic_dir, dataset_dir_name(dataset_name), category)

    if not os.path.exists(synthetic_dir):
        raise FileNotFoundError(f"Synthetic data not found at {synthetic_dir}")

    # Create dataset
    logger.info("\nLoading synthetic data...")
    dataset = SyntheticAnomalyFPFHGeometryDataset(synthetic_dir, category, args, config)

    # Train/val split based on 'split' field
    train_indices = []
    val_indices = []
    for idx in range(len(dataset.data_cache)):
        data = dataset.data_cache[idx]
        split = data.get('split', 'train')
        if split == 'train':
            train_indices.append(idx)
        elif split == 'val':
            val_indices.append(idx)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")

    # Initialize device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")

    # Map dataset type to directory name
    dataset_dir = dataset_dir_name(config.dataset)


    logger.info("\nLoading memory bank...")
    ref_data = load_reference_data_spatial_chamfer(category, dataset=dataset_dir)


    reference_patch_centers = ref_data['spatial_bank']['centers'].to(device)
    reference_patch_fpfh = ref_data['spatial_bank']['fpfh_matrix'].to(device)
    reference_means = ref_data['spatial_bank']['means_matrix'].to(device)
    reference_normals = ref_data['spatial_bank']['normals_matrix'].to(device)

    logger.info(f"  Reference centers: {reference_patch_centers.shape}")
    logger.info(f"  Reference FPFH: {reference_patch_fpfh.shape}")
    logger.info(f"  Reference means: {reference_means.shape}")
    logger.info(f"  Reference normals: {reference_normals.shape}")

    # Create dataloaders with custom collate function for patch extraction
    from functools import partial
    collate_fn = partial(collate_fn_with_patches, num_patches=args.num_patches, patch_size=args.patch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    input_dim = 7  # Memory bank statistics dimension

    model = SigmoidFusion(patch_feature_dim=input_dim, hidden_dim=args.hidden_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel: {total_params:,} parameters (patch_features={input_dim} + scores=2 → MLP input=44)")

    # Loss function: Focal loss for patch classification
    if args.loss_type == 'patch_contrastive':
        def loss_fn(scores, labels):
            return patch_contrastive_loss(
                scores, labels,
                margin=args.loss_margin,
                num_pairs=args.loss_num_pairs
            )
        logger.info(f"Loss: patch_contrastive (margin={args.loss_margin}, pairs={args.loss_num_pairs})")
    elif args.loss_type == 'ranking_reg':
        from utils.loss import ranking_loss_with_regularization
        def loss_fn(scores, labels):
            return ranking_loss_with_regularization(scores, labels, reg_weight=0.01)
        logger.info(f"Loss: ranking_reg (reg_weight=0.01)")
    else:
        loss_fn = get_loss_function(args.loss_type)
        logger.info(f"Loss: {args.loss_type}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"Optimizer: Adam (lr={args.lr}, wd={args.weight_decay})")

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info("=" * 80)

    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # TRAINING - Patch-Level Classification
        model.train()
        train_loss = 0.0
        train_scores_all = []
        train_labels_all = []


        max_train_batches = max(1, int(len(train_loader) * args.epoch_fraction))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", total=max_train_batches)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_train_batches:
                break

            # Get patches extracted by dataloader workers (parallel on CPU!)
            patch_points = batch['patch_points'].to(device)    # (B, num_patches, patch_size, 3)
            patch_fpfh = batch['patch_fpfh'].to(device)        # (B, num_patches, 33)
            patch_centers = batch['patch_centers'].to(device)  # (B, num_patches, 3)
            patch_indices = batch['patch_indices']              # List of (num_patches, patch_size) tensors
            masks = batch['masks']                              # List of (N,) tensors

            B = patch_points.shape[0]


            batch_final_scores = []
            batch_labels = []

            for i in range(B):

                final_scores = model(
                    patch_points[i],           # (P, patch_size, 3)
                    patch_fpfh[i],             # (P, 33)
                    patch_centers[i],          # (P, 3)
                    reference_patch_centers,   # (R, 3)
                    reference_means,           # (R, N, 3)
                    reference_normals,         # (R, N, 3)
                    reference_patch_fpfh,      # (R, N, 33)
                    k_centers=args.k_centers
                )

                # Compute patch labels from mask using patch indices
                mask_i = masks[i].to(device)
                patch_indices_i = patch_indices[i].to(device)
                patch_masks = mask_i[patch_indices_i]  # (num_patches, patch_size)
                patch_labels_sample = (patch_masks.float().mean(dim=1) > 0.5).float()  # (num_patches,)

                batch_final_scores.append(final_scores)
                batch_labels.append(patch_labels_sample)

            # Stack all patches from batch
            final_scores_all = torch.cat(batch_final_scores)  # (B*num_patches,)
            labels_all = torch.cat(batch_labels)  # (B*num_patches,)

            # Compute loss
            loss = loss_fn(final_scores_all, labels_all)


            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track
            train_loss += loss.item()
            train_scores_all.extend(final_scores_all.detach().cpu().numpy())
            train_labels_all.extend(labels_all.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= max(batch_idx + 1, 1)  # Divide by actual number of batches processed
        if len(np.unique(train_labels_all)) > 1:
            train_auroc = roc_auc_score(train_labels_all, train_scores_all)
        else:
            train_auroc = 0.0

        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"  Train: Loss={train_loss:.4f}, AUROC={train_auroc:.4f}")




        should_evaluate = (not args.skip_eval) and ((epoch + 1) % args.eval_every_n_epochs == 0 or (epoch + 1) == args.epochs)

        if should_evaluate:
            logger.info("  Evaluating on test set...")
            test_results = evaluate_on_test_set_with_model(
                model, ref_data, config, category, device, args,
                visualize=args.visualize,
                batch_size=args.eval_batch_size
            )
            logger.info(f"  Test:  I-AUROC={test_results['i_auroc']:.4f}, P-AUROC={test_results['p_auroc']:.4f} " +
                        f"({test_results['num_samples']} samples)")
        else:
            logger.info(f"  Skipping evaluation (eval_every_n_epochs={args.eval_every_n_epochs})")
            test_results = {'i_auroc': 0.0, 'p_auroc': 0.0, 'num_samples': 0}

        # Scheduler step
        scheduler.step(train_loss)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_i_auroc': test_results['i_auroc'],
            'test_p_auroc': test_results['p_auroc'],
        }

        # Save per-epoch checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch_{epoch+1:03d}.pth'))

        # Early stopping
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered!")
            break

        epoch_time = time.time() - epoch_start
        logger.info(f"  Time: {epoch_time:.1f}s")
        logger.info("-" * 80)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train FPFH + Geometry fusion network')

    # Required arguments
    parser.add_argument('--training_config', type=str, required=True,
                        help='Training config file (e.g., configs/training/sigmoid_fusion.yaml)')
    parser.add_argument('--category', type=str, required=True, help='Category name')

    # Dataset config (for data loading)
    parser.add_argument('--dataset_config', type=str, default='configs/real3d-ad/base.yaml',
                        help='Dataset config file (default: configs/real3d-ad/base.yaml)')

    # Optional overrides (command line takes precedence over config file)
    parser.add_argument('--device', type=str, default=None, help='Device (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--visualize', action='store_true', help='Visualize during evaluation')
    parser.add_argument('--max_synthetic_samples', type=int, default=None,
                        help='Optional cap for smoke/debug training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0 for smoke-friendly runs)')
    parser.add_argument('--no-recompute', dest='recompute', action='store_false', help='Use cached test features if available')
    parser.set_defaults(recompute=True)

    args = parser.parse_args()

    # Load training config
    train_config = EasyDict()
    cfg_from_yaml_file(args.training_config, train_config)

    # Load dataset config
    dataset_config = EasyDict()
    cfg_from_yaml_file(args.dataset_config, dataset_config)

    # Merge configs into a single args object
    merged_args = EasyDict()

    # Model config
    merged_args.model_name = train_config.model.name
    merged_args.patch_feature_dim = train_config.model.patch_feature_dim
    merged_args.hidden_dim = train_config.model.hidden_dim

    # Training config
    merged_args.epochs = args.epochs if args.epochs is not None else train_config.training.epochs
    merged_args.batch_size = args.batch_size if args.batch_size is not None else train_config.training.batch_size
    merged_args.lr = train_config.training.learning_rate
    merged_args.weight_decay = train_config.training.weight_decay
    merged_args.loss_type = train_config.training.loss_type
    merged_args.loss_margin = train_config.training.get('loss_margin', 0.5)
    merged_args.loss_num_pairs = train_config.training.get('loss_num_pairs', 100)
    merged_args.early_stopping_patience = train_config.training.early_stopping_patience

    # Patch config
    merged_args.num_patches = train_config.patches.num_patches
    merged_args.num_patches2 = train_config.patches.num_patches_test
    merged_args.patch_size = train_config.patches.patch_size
    merged_args.k_centers = train_config.patches.k_centers

    # Data config
    merged_args.synthetic_dir = train_config.data.get('synthetic_dir', 'synthetic_data')
    merged_args.synthetic_config = train_config.data.synthetic_config

    # Evaluation config
    merged_args.eval_batch_size = train_config.evaluation.get('batch_size', 8)
    merged_args.eval_every_n_epochs = train_config.evaluation.get('eval_every_n_epochs', 1)
    merged_args.skip_eval = train_config.evaluation.get('skip_eval', False)

    # Other
    merged_args.seed = train_config.seed
    merged_args.device = args.device if args.device is not None else 'cuda:0'
    merged_args.visualize = args.visualize
    merged_args.recompute = args.recompute
    merged_args.epoch_fraction = 1.0
    merged_args.max_synthetic_samples = args.max_synthetic_samples
    merged_args.num_workers = args.num_workers

    # Pass through required args
    merged_args.category = args.category
    merged_args.training_config_path = args.training_config
    merged_args.dataset_config_path = args.dataset_config

    # Attach dataset config
    merged_args.dataset_config = dataset_config

    # Run training
    train_fusion_network(merged_args)


if __name__ == '__main__':
    main()
