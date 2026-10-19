"""
3-Way Comparison: FPFH vs Geometry vs Learned Fusion Anomaly Detection

This script visualizes the difference between three anomaly scoring methods:
- FPFH: Pure FPFH features (Fast Point Feature Histograms)
- Geometry: Pure geometry (mean distance + normal angle)
- Learned Fusion: Trained neural network fusion model from logs/

Visualization modes:
- Default: 3-way comparison with synchronized camera views
- --learned_only: Single view showing only learned fusion results

Usage:
    # 3-way comparison (default)
    python compare_geometry_fpfh.py configs/real3d-ad/base.yaml \
        --category bagel --test_file 001.pcd \
        --num_patches 3000 --patch_size 128 \
        --epoch 10

    # Learned fusion only
    python compare_geometry_fpfh.py configs/real3d-ad/base.yaml \
        --category bagel --test_file 001.pcd \
        --learned_only

Arguments:
    - config: Dataset config file (required)
    - --category: Category name (required)
    - --test_file: Specific test file to visualize (optional, auto-selects if not provided)
    - --num_patches: Number of patches (default: 3000)
    - --patch_size: Points per patch (default: 128)
    - --k_centers: Nearest centers to check (default: 1)
    - --max_points_fpfh: Max points for FPFH (default: 150000)
    - --registration_method: Registration method (default: 'parenet')
    - --checkpoint_path: Path to checkpoint (optional, auto-detects if not provided)
    - --epoch: Epoch to load (default: 10)
    - --learned_only: Show only learned fusion (skip 3-way comparison)
"""

import os
import sys
import argparse
import numpy as np
import torch
from easydict import EasyDict

# Import configuration utilities
from utils.config import cfg_from_yaml_file
from utils.misc import seed_all
from utils.dataset import all_real3dad_cates, all_shapenetad_cates

# Import data loading utilities
from utils.data_loading import (
    load_reference_data_spatial_chamfer,
    find_test_samples,
    load_test_pointcloud
)

# Import sampling utilities
from utils.sampling import extract_patches_gpu_fps
import fpsample

# Import FPFH computation
from utils.fpfh_utils import compute_fpfh_features

# Import registration functions
from registration.methods.parenet_torch import parenet_torch

# Import visualization
import pyvista as pv
from registration.pyvista_visualizer import PyVistaAnomalyVisualizer

# Import core functions from test_manual_fusion_fpfhchamfer.py
# We'll copy/paste the necessary functions here to keep the script self-contained

def apply_transformation_torch(points, transform):
    """Apply 4x4 transformation matrix to torch tensor points on GPU."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    points_transformed = (points @ R.T) + t
    return points_transformed


def register_pointcloud_cached(test_points, reference_points, cache_path, use_cache=True, registration_method='default'):
    """
    Register test pointcloud to reference with caching support.

    Args:
        test_points: Test point cloud torch tensor (N, 3) on GPU
        reference_points: Reference point cloud torch tensor (M, 3) on GPU
        cache_path: Path to cache file (.pkl)
        use_cache: If False, always perform registration
        registration_method: 'default' (9dof), 'patch_dropout', or 'parenet'

    Returns:
        tuple: (transform_torch, test_aligned_torch) - both on GPU
    """
    import pickle
    device = test_points.device

    if use_cache and os.path.exists(cache_path):
        # Load cached transform
        with open(cache_path, 'rb') as f:
            transform = pickle.load(f)
        transform = transform.to(device)
        test_aligned = apply_transformation_torch(test_points, transform)
    else:

        transform = parenet_torch(test_points, reference_points)

        test_aligned = apply_transformation_torch(test_points, transform)

        # Save transform to cache
        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(transform, f)

    return transform, test_aligned


def sample_points(points, max_points):
    """Sample points using FPS if needed."""
    if len(points) > max_points:
        return fpsample.bucket_fps_kdline_sampling(points, max_points, h=5)
    return None


def compute_patch_normals(patch_points):
    """Compute surface normal for each patch using PCA (smallest eigenvector)."""
    num_patches, patch_size, _ = patch_points.shape

    # Center each patch
    patch_centers = patch_points.mean(dim=1, keepdim=True)
    centered_patches = patch_points - patch_centers

    # Compute covariance matrix
    cov = torch.bmm(centered_patches.transpose(1, 2), centered_patches) / patch_size

    # Compute eigenvalues/eigenvectors using SVD
    try:
        U, S, V = torch.svd(cov)
        normals = U[:, :, 2]  # Normal is eigenvector with smallest eigenvalue
    except:
        # Fallback: cross product of two vectors
        v1 = centered_patches[:, 0, :]
        v2 = centered_patches[:, patch_size//2, :]
        normals = torch.cross(v1, v2, dim=1)

    # Normalize
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-6)

    return normals


def compute_mean_normal_scores(
    patch_points_batch,
    patch_center_batch,
    reference_patch_centers,
    reference_patch_points,
    k_centers=10
):
    """
    Compute geometry-based anomaly scores using patch mean + normal comparison.

    Returns:
        score_geometry: (P,) Geometry branch scores
    """
    P = patch_points_batch.shape[0]
    R, N, patch_size, _ = reference_patch_points.shape

    # Compute test patch means and normals
    test_means = patch_points_batch.mean(dim=1)  # (P, 3)
    test_normals = compute_patch_normals(patch_points_batch)  # (P, 3)

    # Precompute all reference patch means and normals
    ref_patches_flat = reference_patch_points.reshape(R * N, patch_size, 3)
    ref_means = ref_patches_flat.mean(dim=1).reshape(R, N, 3)  # (R, N, 3)
    ref_normals = compute_patch_normals(ref_patches_flat).reshape(R, N, 3)  # (R, N, 3)

    # Find k nearest centers in 3D space
    k = min(k_centers, R)
    center_dists = torch.cdist(patch_center_batch, reference_patch_centers)  # (P, R)
    _, nearest_indices = torch.topk(center_dists, k, dim=1, largest=False)  # (P, k)

    # For each test patch, compute distance to nearest reference patches
    score_geometry_list = []

    for p in range(P):
        nearest_center_indices = nearest_indices[p]  # (k,)

        # Get means and normals from k nearest centers
        nearest_ref_means = ref_means[nearest_center_indices]  # (k, N, 3)
        nearest_ref_normals = ref_normals[nearest_center_indices]  # (k, N, 3)

        # Flatten to (k*N, 3)
        nearest_ref_means_flat = nearest_ref_means.reshape(k * N, 3)
        nearest_ref_normals_flat = nearest_ref_normals.reshape(k * N, 3)

        # Compute mean distance (L2 distance between centroids)
        mean_dists = torch.norm(nearest_ref_means_flat - test_means[p].unsqueeze(0), dim=1)

        # Compute normal angle (1 - |cos(angle)|)
        cos_angles = torch.abs(torch.sum(nearest_ref_normals_flat * test_normals[p].unsqueeze(0), dim=1))
        normal_diffs = 1.0 - cos_angles

        # Combine: weighted sum
        combined_scores = mean_dists * 0.5 + normal_diffs * 0.5

        min_score = torch.min(combined_scores)
        score_geometry_list.append(min_score)

        del nearest_ref_means, nearest_ref_normals, mean_dists, normal_diffs, combined_scores

    score_geometry = torch.stack(score_geometry_list)
    return score_geometry


def compute_fpfh_scores(
    patch_fpfh_batch,
    patch_center_batch,
    reference_patch_centers,
    reference_patch_fpfh,
    k_centers=10
):
    """
    Compute FPFH-based anomaly scores.

    Returns:
        score_fpfh: (P,) FPFH branch scores
    """
    P = patch_fpfh_batch.shape[0]
    R, N, _ = reference_patch_fpfh.shape
    device = patch_fpfh_batch.device

    # Find k nearest centers in 3D space
    k = min(k_centers, R)
    center_dists = torch.cdist(patch_center_batch, reference_patch_centers)
    _, nearest_indices = torch.topk(center_dists, k, dim=1, largest=False)

    # Process k centers one at a time (memory-efficient)
    min_dists_fpfh = torch.full((P,), float('inf'), device=device)

    for ki in range(k):
        center_idx = nearest_indices[:, ki]
        center_features = reference_patch_fpfh[center_idx]  # (P, N, 33)

        # Compute distances
        diffs = patch_fpfh_batch.unsqueeze(1) - center_features
        dists = torch.abs(diffs).mean(dim=2)  # (P, N)

        # Find min for this center
        local_min, _ = dists.min(dim=1)
        min_dists_fpfh = torch.min(min_dists_fpfh, local_min)

        del center_features, diffs, dists

    return min_dists_fpfh


def compute_scores(patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, ref_data, fusion_p, k_centers=10):
    """
    Compute anomaly scores using spatial FPFH + Geometry memory bank.

    Args:
        fusion_p: Fusion weight (0=Geometry, 1=FPFH)

    Returns:
        patch_scores_gpu: (num_patches,) anomaly scores on GPU
    """
    # Get reference data from spatial bank
    reference_patch_centers_gpu = ref_data['spatial_bank']['centers']
    reference_patch_fpfh_gpu = ref_data['spatial_bank']['fpfh_matrix']
    reference_patch_points_gpu = ref_data['spatial_bank']['points_matrix']

    # Compute geometry scores
    score_geometry = compute_mean_normal_scores(
        patch_points_gpu,
        patch_centers_gpu,
        reference_patch_centers_gpu,
        reference_patch_points_gpu,
        k_centers=k_centers
    )

    # Compute FPFH scores
    score_fpfh = compute_fpfh_scores(
        patch_fpfh_gpu,
        patch_centers_gpu,
        reference_patch_centers_gpu,
        reference_patch_fpfh_gpu,
        k_centers=k_centers
    )

    # Fuse scores
    patch_scores_gpu = score_geometry * (1 - fusion_p) + score_fpfh * fusion_p

    return patch_scores_gpu


def remap_patch_scores_to_points(patch_scores, patch_indices, num_points, point_cloud=None):
    """
    Remap patch scores to point scores using index-based mapping.

    Args:
        patch_scores: (num_patches,) numpy array of patch scores
        patch_indices: list of (patch_size,) arrays - KNN indices for each patch
        num_points: total number of points
        point_cloud: (num_points, 3) torch tensor on GPU (optional, for uncovered points)

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

        device = point_cloud.device
        covered_points = point_cloud[covered_indices]
        uncovered_points = point_cloud[uncovered_indices]

        # PyKeOps KNN
        X_i = LazyTensor(uncovered_points[:, None, :])
        X_j = LazyTensor(covered_points[None, :, :])
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        nn_indices = D_ij.argKmin(1, dim=1).squeeze()

        point_scores[uncovered_indices] = point_scores[covered_indices[nn_indices.cpu().numpy()]]

    return point_scores


def remap_sampled_to_full(point_scores_sampled, sample_indices, num_points_full, point_cloud_full):
    """
    Remap scores from sampled points to full point cloud.

    Args:
        point_scores_sampled: (num_sampled,) scores for sampled points
        sample_indices: (num_sampled,) indices of sampled points, or None if no sampling
        num_points_full: total number of points in full cloud
        point_cloud_full: (num_points_full, 3) full point cloud coordinates

    Returns:
        point_scores_full: (num_points_full,) scores for all points
    """
    if sample_indices is None:
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


def find_latest_checkpoint(category, epoch=10):
    """
    Auto-detect latest training run and load specified epoch checkpoint.

    Searches logs/{category}/ for most recent training run directory,
    returns path to epoch_{epoch:03d}.pth checkpoint.

    Args:
        category: Category name (e.g., 'bagel', 'airplane')
        epoch: Epoch number to load (default: 10)

    Returns:
        Path to checkpoint file or None if not found
    """
    logs_dir = f"logs/{category}"

    if not os.path.exists(logs_dir):
        return None

    # Get all training run directories, sorted by timestamp (newest first)
    try:
        run_dirs = sorted(
            [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))],
            reverse=True
        )
    except Exception:
        return None

    if not run_dirs:
        return None

    # Use latest run
    latest_run = run_dirs[0]
    checkpoint_path = os.path.join(logs_dir, latest_run, "checkpoints", f"epoch_{epoch:03d}.pth")

    return checkpoint_path if os.path.exists(checkpoint_path) else None


def load_fusion_model(checkpoint_path, device='cuda'):
    """
    Load trained fusion model from checkpoint and training config.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on (default: 'cuda')

    Returns:
        tuple: (model, model_name, training_config) - Loaded model in eval mode, architecture name, and training config

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted
        ValueError: If model architecture not supported
    """
    import yaml
    from pathlib import Path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    if 'model_state_dict' not in checkpoint:
        raise ValueError(f"Invalid checkpoint: missing 'model_state_dict'")

    # Detect model architecture
    model_name = checkpoint.get('model_name', 'SigmoidFusion')

    # Import and instantiate model
    if model_name == 'SigmoidFusion':
        from models.fusion_models import SigmoidFusion
        model = SigmoidFusion()
    elif model_name == 'HierarchicalFusion':
        from models.fusion_models import HierarchicalFusion
        model = HierarchicalFusion()
    elif model_name == 'ImprovedAttentionFusionV2':
        from models.fusion_models import ImprovedAttentionFusionV2
        model = ImprovedAttentionFusionV2()
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load training config from log directory
    checkpoint_dir = Path(checkpoint_path).parent
    log_dir = checkpoint_dir.parent
    training_config_path = log_dir / 'training_config.yaml'

    training_config = {}
    if training_config_path.exists():
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        print(f"    Loaded training config from {training_config_path}")
    else:
        print(f"    Warning: Training config not found at {training_config_path}, using defaults")

    return model, model_name, training_config


def compute_learned_fusion_scores(
    model,
    model_name,
    patch_points_gpu,
    patch_fpfh_gpu,
    patch_centers_gpu,
    ref_data,
    k_centers=10
):
    """
    Run model inference to get anomaly scores from learned fusion.

    Args:
        model: Loaded fusion model in eval mode
        model_name: Architecture name (determines interface)
        patch_points_gpu: (num_patches, patch_size, 3) patch points on GPU
        patch_fpfh_gpu: (num_patches, 33) FPFH features on GPU
        patch_centers_gpu: (num_patches, 3) patch centers on GPU
        ref_data: Memory bank dict with spatial_bank
        k_centers: Number of nearest centers to check (default: 10)

    Returns:
        scores: (num_patches,) anomaly scores on GPU
    """
    # Extract reference data from memory bank
    reference_patch_centers = ref_data['spatial_bank']['centers']
    reference_patch_fpfh = ref_data['spatial_bank']['fpfh_matrix']
    reference_patch_points = ref_data['spatial_bank']['points_matrix']

    # Compute reference means and normals from points_matrix
    R, N, patch_size, _ = reference_patch_points.shape
    ref_patches_flat = reference_patch_points.reshape(R * N, patch_size, 3)

    # Compute means: (R*N, patch_size, 3) -> (R*N, 3) -> (R, N, 3)
    reference_means = ref_patches_flat.mean(dim=1).reshape(R, N, 3)

    # Compute normals using existing function: (R*N, patch_size, 3) -> (R*N, 3) -> (R, N, 3)
    reference_normals = compute_patch_normals(ref_patches_flat).reshape(R, N, 3)

    # Run model inference (no gradient computation)
    with torch.no_grad():
        if model_name in ['SigmoidFusion', 'HierarchicalFusion']:
            scores = model(
                patch_points_gpu,
                patch_fpfh_gpu,
                patch_centers_gpu,
                reference_patch_centers,
                reference_means,
                reference_normals,
                reference_patch_fpfh,
                k_centers=k_centers
            )
        elif model_name == 'ImprovedAttentionFusionV2':
            # Different interface for attention-based models
            raise NotImplementedError(
                f"ImprovedAttentionFusionV2 inference not yet implemented. "
                f"Use SigmoidFusion or HierarchicalFusion instead."
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    return scores


def visualize_learned_only(
    points,
    scores_learned,
    model_name,
    epoch,
    title="Learned Fusion",
    gt_labels=None,
    camera_pos=None
):
    """
    Visualize only learned fusion scores using PyVistaAnomalyVisualizer.

    Args:
        points: Point cloud numpy array (N, 3)
        scores_learned: Learned fusion scores (N,)
        model_name: Name of the loaded fusion model
        epoch: Epoch number of the loaded model
        title: Window title
        gt_labels: Optional ground truth labels (N,)
        camera_pos: Optional camera position (x, y, z) tuple/list
    """
    print(f"\n[Interactive PyVista Viewer]")
    print(f"  Keyboard shortcuts:")
    print(f"    • Click: Pick points for values")
    print(f"    • n/p: Next/Previous colormap")
    print(f"    • t/y: Scale down/up • u: Reset scale")
    print(f"    • l/m: Point size • f: Toggle axes • h: Help • q: Quit\n")

    # Use PyVistaAnomalyVisualizer for interactive visualization
    visualizer = PyVistaAnomalyVisualizer(
        colormap='jet',
        interactive=True,
        save_screenshots=False,
        point_size=9.0,
        show_scalar_bar=True,
        window_size=(1200, 900),
        show_boundary=False,
        show_reference_frame=True,
        camera_position=camera_pos
    )

    sample_name = f"{title} ({model_name}, epoch {epoch})"
    visualizer.visualize(points, scores_learned, sample_name=sample_name, gt_labels=gt_labels)


def visualize_three_way_comparison(
    points,
    scores_fpfh,
    scores_geometry,
    scores_learned,
    model_name,
    epoch,
    title="Comparison",
    gt_labels=None
):
    """
    Visualize 3-way comparison: FPFH vs Geometry vs Learned Fusion.

    Args:
        points: Point cloud numpy array (N, 3)
        scores_fpfh: FPFH scores (N,)
        scores_geometry: Geometry scores (mean + normal) (N,)
        scores_learned: Learned fusion scores (N,)
        model_name: Name of the loaded fusion model
        epoch: Epoch number of the loaded model
        title: Window title
        gt_labels: Optional ground truth labels (N,)
    """
    # Create 1x3 horizontal layout
    plotter = pv.Plotter(shape=(1, 3), window_size=(3600, 900))

    # Normalize scores independently for each view
    scores_fpfh_norm = (scores_fpfh - scores_fpfh.min()) / (scores_fpfh.max() - scores_fpfh.min() + 1e-6)
    scores_geometry_norm = (scores_geometry - scores_geometry.min()) / (scores_geometry.max() - scores_geometry.min() + 1e-6)
    scores_learned_norm = (scores_learned - scores_learned.min()) / (scores_learned.max() - scores_learned.min() + 1e-6)

    # Left subplot: FPFH
    plotter.subplot(0, 0)
    plotter.add_text(
        f"FPFH (features)\nRange: {scores_fpfh.min():.6f} - {scores_fpfh.max():.6f}",
        font_size=12, position='upper_edge'
    )

    cloud_fpfh = pv.PolyData(points)
    cloud_fpfh['scores'] = scores_fpfh_norm
    if gt_labels is not None:
        cloud_fpfh['gt_label'] = gt_labels

    plotter.add_mesh(
        cloud_fpfh,
        scalars='scores',
        cmap='jet',
        point_size=8.0,
        render_points_as_spheres=True,
        scalar_bar_args={'title': 'FPFH Score', 'n_labels': 5}
    )
    plotter.add_axes()
    plotter.camera_position = 'iso'

    # Middle subplot: Geometry (Mean + Normal)
    plotter.subplot(0, 1)
    plotter.add_text(
        f"Geometry (Mean+Normal)\nRange: {scores_geometry.min():.6f} - {scores_geometry.max():.6f}",
        font_size=12, position='upper_edge'
    )

    cloud_geometry = pv.PolyData(points)
    cloud_geometry['scores'] = scores_geometry_norm
    if gt_labels is not None:
        cloud_geometry['gt_label'] = gt_labels

    plotter.add_mesh(
        cloud_geometry,
        scalars='scores',
        cmap='jet',
        point_size=8.0,
        render_points_as_spheres=True,
        scalar_bar_args={'title': 'Geometry Score', 'n_labels': 5}
    )
    plotter.add_axes()
    plotter.camera_position = 'iso'

    # Right subplot: Learned Fusion
    plotter.subplot(0, 2)
    plotter.add_text(
        f"Learned Fusion ({model_name}, epoch {epoch})\nRange: {scores_learned.min():.6f} - {scores_learned.max():.6f}",
        font_size=12, position='upper_edge'
    )

    cloud_learned = pv.PolyData(points)
    cloud_learned['scores'] = scores_learned_norm
    if gt_labels is not None:
        cloud_learned['gt_label'] = gt_labels

    plotter.add_mesh(
        cloud_learned,
        scalars='scores',
        cmap='jet',
        point_size=8.0,
        render_points_as_spheres=True,
        scalar_bar_args={'title': 'Learned Score', 'n_labels': 5}
    )
    plotter.add_axes()
    plotter.camera_position = 'iso'

    # Link cameras for synchronized viewing
    plotter.link_views()

    # Add overall title
    plotter.add_text(
        f"{title}\n3-Way Comparison: FPFH | Geometry | Learned Fusion | Cameras linked for synchronized rotation/zoom",
        position='lower_edge', font_size=10
    )

    plotter.show()


def main(args):
    """Main pipeline for side-by-side comparison visualization."""
    # Load config and setup
    config = EasyDict()
    cfg_from_yaml_file(args.config, config)

    category = args.category
    dataset_path = config.dataset_path
    dataset_type = config.dataset

    seed_all(args.seed)

    # Map dataset type to directory name
    dataset_dir_map = {
        'Real3DAD': 'real3d-ad',
        'ShapeNetAD': 'shapenet-ad',
    }
    dataset_dir = dataset_dir_map.get(dataset_type, dataset_type.lower())

    print(f"\n{'='*80}")
    print(f"3-Way Comparison: FPFH vs Geometry vs Learned Fusion")
    print(f"Category: {category}")
    print(f"Dataset: {dataset_type}")
    print(f"{'='*80}\n")

    # Load reference data with spatial memory bank
    print("Loading reference data...")
    ref_data = load_reference_data_spatial_chamfer(category, dataset=dataset_dir)

    # Compute voxel size for FPFH
    diameter = np.linalg.norm(ref_data['points'].max(axis=0) - ref_data['points'].min(axis=0))
    voxel_size = diameter * 0.05

    # Find test samples
    if args.test_file is None:
        # Auto-select: find all test samples and process them one by one
        test_samples = find_test_samples(dataset_path, category, dataset_type, None)
        if len(test_samples) == 0:
            print(f"No test samples found for category {category}")
            return
        print(f"Auto-selecting all test files: {len(test_samples)} files found")
    else:
        # User specified a test file
        test_samples = find_test_samples(dataset_path, category, dataset_type, args.test_file)
        if len(test_samples) == 0:
            print(f"No test samples found for category {category}, file {args.test_file}")
            return

    # Move memory bank to GPU
    device = torch.device('cuda')
    ref_data['spatial_bank']['centers'] = ref_data['spatial_bank']['centers'].to(device)
    ref_data['spatial_bank']['fpfh_matrix'] = ref_data['spatial_bank']['fpfh_matrix'].to(device)
    ref_data['spatial_bank']['points_matrix'] = ref_data['spatial_bank']['points_matrix'].to(device)

    # Convert reference points to GPU
    ref_points_gpu = torch.from_numpy(ref_data['points']).float().cuda()

    # Process all test samples one by one
    for idx, (subdir, test_file, test_file_path) in enumerate(test_samples, 1):
        print(f"\n{'='*80}")
        print(f"Processing test sample {idx}/{len(test_samples)}: {subdir}/{test_file}")
        print(f"{'='*80}")

        # Load test point cloud
        test_data = load_test_pointcloud(test_file_path, dataset_type, subdir, is_evaluate=True)
        points_normalized = test_data['points_normalized']
        gt_mask = test_data['gt_mask']

        # Sample points if needed
        sample_indices = sample_points(points_normalized, args.max_points_fpfh)

        # Convert to GPU
        test_points_gpu = torch.from_numpy(points_normalized).float().cuda()

        # Apply sampling
        if sample_indices is not None:
            test_sampled = test_points_gpu[sample_indices]
            print(f"  Sampled: {len(test_sampled)}/{len(test_points_gpu)} points")
        else:
            test_sampled = test_points_gpu

        # Registration
        cache_path = f'test_reg/{dataset_dir}/{category}/{subdir}/{test_file.replace(".pcd", ".pkl")}'
        print(f"  Registration: ", end='')
        transform, test_aligned = register_pointcloud_cached(
            test_sampled, ref_points_gpu, cache_path,
            use_cache=True,
            registration_method=args.registration_method
        )
        print("Done")

        # Compute FPFH
        print(f"  Computing FPFH...", end='')
        test_fpfh = compute_fpfh_features(test_aligned, voxel_size)
        print(f" Done")

        # Extract patches
        print(f"  Extracting patches...", end='')
        patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, patch_indices = extract_patches_gpu_fps(
            test_aligned, test_fpfh, args.num_patches, args.patch_size
        )
        print(f" {len(patch_points_gpu)} patches")

        # Convert patch indices to CPU
        patch_indices_cpu = patch_indices.cpu().numpy()

        # Get reference data from memory bank
        reference_patch_centers_gpu = ref_data['spatial_bank']['centers']
        reference_patch_fpfh_gpu = ref_data['spatial_bank']['fpfh_matrix']
        reference_patch_points_gpu = ref_data['spatial_bank']['points_matrix']

        # Compute scores for all three methods
        print(f"  Computing FPFH scores...", end='')
        patch_scores_fpfh = compute_fpfh_scores(
            patch_fpfh_gpu, patch_centers_gpu,
            reference_patch_centers_gpu, reference_patch_fpfh_gpu,
            k_centers=args.k_centers
        )
        print(f" Range: {patch_scores_fpfh.min().item():.6f} - {patch_scores_fpfh.max().item():.6f}")

        print(f"  Computing geometry scores (mean + normal)...", end='')
        patch_scores_geometry = compute_mean_normal_scores(
            patch_points_gpu, patch_centers_gpu,
            reference_patch_centers_gpu, reference_patch_points_gpu,
            k_centers=args.k_centers
        )
        print(f" Range: {patch_scores_geometry.min().item():.6f} - {patch_scores_geometry.max().item():.6f}")

        # Load and apply learned fusion model
        print(f"  Loading learned fusion model...")
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = find_latest_checkpoint(category, args.epoch)

        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found for {category} epoch {args.epoch}")

        print(f"    Checkpoint: {checkpoint_path}")
        model, model_name, training_config = load_fusion_model(checkpoint_path, device='cuda')

        # Use training config parameters if available, otherwise fall back to args
        if training_config and 'patches' in training_config:
            k_centers_to_use = training_config['patches'].get('k_centers', args.k_centers)
            print(f"    Using k_centers={k_centers_to_use} from training config")
        else:
            k_centers_to_use = args.k_centers
            print(f"    Using k_centers={k_centers_to_use} from command-line args")

        print(f"  Computing learned fusion scores ({model_name})...", end='')
        patch_scores_learned = compute_learned_fusion_scores(
            model, model_name,
            patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu,
            ref_data, k_centers=k_centers_to_use
        )
        print(f" Range: {patch_scores_learned.min().item():.6f} - {patch_scores_learned.max().item():.6f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # Remap patch scores to point scores
        print(f"  Remapping scores to points...")

        # Learned fusion scores (always needed)
        point_scores_learned_sampled = remap_patch_scores_to_points(
            patch_scores_learned.cpu().numpy(),
            patch_indices_cpu,
            len(test_aligned),
            test_aligned
        )

        point_scores_learned_full = remap_sampled_to_full(
            point_scores_learned_sampled,
            sample_indices,
            len(points_normalized),
            points_normalized
        )

        # Visualize based on mode
        if args.learned_only:
            # Single view: learned fusion only
            print(f"\nDisplaying learned fusion only...")
            print(f"  Model: {model_name}, epoch {args.epoch}\n")

            visualize_learned_only(
                points_normalized,
                point_scores_learned_full,
                model_name=model_name,
                epoch=args.epoch,
                title=f"{category} - {subdir}/{test_file}",
                gt_labels=gt_mask,
                camera_pos=args.cam
            )
        else:
            # 3-way comparison: FPFH + Geometry + Learned
            # FPFH scores
            point_scores_fpfh_sampled = remap_patch_scores_to_points(
                patch_scores_fpfh.cpu().numpy(),
                patch_indices_cpu,
                len(test_aligned),
                test_aligned
            )

            point_scores_fpfh_full = remap_sampled_to_full(
                point_scores_fpfh_sampled,
                sample_indices,
                len(points_normalized),
                points_normalized
            )

            # Geometry scores
            point_scores_geometry_sampled = remap_patch_scores_to_points(
                patch_scores_geometry.cpu().numpy(),
                patch_indices_cpu,
                len(test_aligned),
                test_aligned
            )

            point_scores_geometry_full = remap_sampled_to_full(
                point_scores_geometry_sampled,
                sample_indices,
                len(points_normalized),
                points_normalized
            )

            print(f"\nDisplaying 3-way comparison...")
            print(f"  Left: FPFH (feature descriptors)")
            print(f"  Middle: Geometry (mean + normal angle)")
            print(f"  Right: Learned Fusion ({model_name}, epoch {args.epoch})")
            print(f"  Cameras are linked for synchronized rotation/zoom\n")

            visualize_three_way_comparison(
                points_normalized,
                point_scores_fpfh_full,
                point_scores_geometry_full,
                point_scores_learned_full,
                model_name=model_name,
                epoch=args.epoch,
                title=f"{category} - {subdir}/{test_file}",
                gt_labels=gt_mask
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="3-way comparison: FPFH vs Geometry vs Learned Fusion anomaly detection"
    )
    parser.add_argument('config', help='Dataset config file')
    parser.add_argument('--category', type=str, required=True, help='Category name')
    parser.add_argument('--test_file', type=str, default=None, help='Specific test file to visualize (auto-selects first if not specified)')
    parser.add_argument('--num_patches', type=int, default=3000, help='Number of patches')
    parser.add_argument('--patch_size', type=int, default=128, help='Points per patch')
    parser.add_argument('--k_centers', type=int, default=1, help='Number of nearest centers to check')
    parser.add_argument('--max_points_fpfh', type=int, default=150000, help='Maximum points for FPFH computation')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--registration_method', type=str, default='parenet',
                        choices=['default', 'patch_dropout', 'parenet'],
                        help='Registration method: default (adam 9dof), patch_dropout, or parenet')

    # Learned fusion arguments
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint (auto-detect latest run if not specified)')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Epoch to load (default: 10)')

    # Visualization mode
    parser.add_argument('--learned_only', action='store_true',
                        help='Visualize only learned fusion (skip FPFH and geometry comparison)')

    # Camera position
    parser.add_argument('--cam', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
                        help='Camera position (x, y, z), e.g., --cam 3.14 -4.04 -1.32')

    args = parser.parse_args()
    main(args)
