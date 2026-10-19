"""
Simple FPFH-based anomaly detection with patch averaging.

Pipeline:
1. Training: Sample points → Compute FPFH → Create patches → Average FPFH per patch → Memory bank
2. Test: Sample 150k points → Create 2048 patches of 128 points → Average FPFH per patch → k-NN → Score

Usage:
    python test_fpfh_simple.py configs/real3d-ad/base.yaml --category airplane --test_file 363_bulge.pcd
"""
import os
import sys
import argparse
import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict
from utils.config import cfg_from_yaml_file
from utils.misc import seed_all
from utils.data_loading import normalize_shape_bbox, load_test_pointcloud, find_test_samples
from utils.metrics import compute_metrics
from registration.pyvista_visualizer import PyVistaAnomalyVisualizer
from tqdm import tqdm
import fpsample

from utils.fpfh_utils import compute_fpfh_features


def load_train_samples(dataset_path, category, dataset_type):
    """Load all training samples for a category."""
    if dataset_type == 'Real3DAD':
        train_dir = os.path.join(dataset_path, category, 'train')
    elif dataset_type == 'ShapeNetAD':
        train_dir = os.path.join(dataset_path, 'train', category)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Get all .pcd files
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.pcd')]

    train_paths = [os.path.join(train_dir, f) for f in train_files]
    return train_paths


def load_pointcloud(path, dataset_type):
    """Load and normalize a point cloud."""


    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float32)

    points_normalized = normalize_shape_bbox(points)
    return points_normalized


def extract_patches_fps_knn(points_gpu, fpfh_gpu, num_patches, patch_size):
    """
    Extract patches using FPS for centers and KNN for patch points.

    Args:
        points_gpu: (N, 3) points on GPU
        fpfh_gpu: (N, 33) FPFH features on GPU
        num_patches: Number of patches to extract
        patch_size: Points per patch

    Returns:
        patch_fpfh_avg: (num_patches, 33) averaged FPFH per patch
        patch_centers: (num_patches, 3) patch center coordinates
        patch_indices: (num_patches, patch_size) indices of points in each patch
    """
    N = len(points_gpu)

    # FPS to select patch centers
    num_centers = min(num_patches, N)
    centers_indices = fpsample.bucket_fps_kdline_sampling(points_gpu.cpu().numpy(), num_centers, h=5)
    centers_indices_gpu = torch.from_numpy(centers_indices).long().cuda()
    patch_centers = points_gpu[centers_indices_gpu]  # (num_patches, 3)

    # For each center, find KNN points to form patch
    patch_indices_list = []
    patch_fpfh_list = []

    for i in range(num_centers):
        center = patch_centers[i:i+1]  # (1, 3)

        # Compute distances to all points
        dists = torch.norm(points_gpu - center, dim=1)  # (N,)

        # Get k nearest neighbors
        k = min(patch_size, N)
        _, knn_idx = torch.topk(dists, k, largest=False)  # (k,)

        # Get FPFH features for this patch
        patch_fpfh = fpfh_gpu[knn_idx]  # (k, 33)

        # Average FPFH features across the patch
        patch_fpfh_avg = patch_fpfh.mean(dim=0)  # (33,)

        patch_indices_list.append(knn_idx)
        patch_fpfh_list.append(patch_fpfh_avg)

    patch_fpfh_avg = torch.stack(patch_fpfh_list, dim=0)  # (num_patches, 33)
    patch_indices = torch.stack(patch_indices_list, dim=0)  # (num_patches, patch_size)

    return patch_fpfh_avg, patch_centers, patch_indices


def build_memory_bank(train_paths, dataset_type, voxel_size, num_patches=2048, patch_size=128):
    """
    Build FPFH memory bank from training samples using patch averaging.

    Args:
        train_paths: List of training file paths
        dataset_type: Dataset type
        voxel_size: Voxel size for FPFH
        num_patches: Number of patches per sample
        patch_size: Points per patch

    Returns:
        dict with averaged FPFH features per patch
    """
    all_patch_fpfh = []

    print(f"Building memory bank from {len(train_paths)} training samples...")

    for path in tqdm(train_paths, desc="Processing training samples"):
        # Load
        points = load_pointcloud(path, dataset_type)

        # Sample 150k points
        if len(points) > 150000:
            indices = fpsample.bucket_fps_kdline_sampling(points, 150000, h=5)
            points = points[indices]

        # Convert to GPU
        points_gpu = torch.from_numpy(points).float().cuda()

        # Compute FPFH
        fpfh_gpu = compute_fpfh_features(points_gpu, voxel_size)

        # Extract patches and average FPFH
        patch_fpfh_avg, _, _ = extract_patches_fps_knn(points_gpu, fpfh_gpu, num_patches, patch_size)

        all_patch_fpfh.append(patch_fpfh_avg)

    # Concatenate all patches from all training samples
    memory_bank_fpfh = torch.cat(all_patch_fpfh, dim=0)  # (N_train * num_patches, 33)

    print(f"✓ Memory bank built: {len(memory_bank_fpfh)} patch features")

    return {
        'fpfh': memory_bank_fpfh  # Keep on GPU
    }


def compute_anomaly_scores(test_patch_fpfh_gpu, memory_bank_fpfh_gpu, k=1):
    """
    Compute anomaly scores as k-NN distance in averaged FPFH space.

    Args:
        test_patch_fpfh_gpu: (num_patches, 33) averaged test patch FPFH on GPU
        memory_bank_fpfh_gpu: (N_bank, 33) memory bank FPFH on GPU
        k: Number of nearest neighbors

    Returns:
        scores: (num_patches,) anomaly scores per patch
    """
    N_test = test_patch_fpfh_gpu.shape[0]
    N_bank = memory_bank_fpfh_gpu.shape[0]

    # Compute pairwise L1 distances
    diffs = test_patch_fpfh_gpu.unsqueeze(1) - memory_bank_fpfh_gpu.unsqueeze(0)  # (N_test, N_bank, 33)
    distances = torch.abs(diffs).mean(dim=2)  # (N_test, N_bank)

    # Find k nearest neighbors
    k = min(k, N_bank)
    knn_dists, _ = torch.topk(distances, k, dim=1, largest=False)

    # Anomaly score = mean distance to k-NN
    scores = knn_dists.mean(dim=1)  # (N_test,)

    return scores


def remap_patch_scores_to_points(patch_scores, patch_indices, num_points):
    """
    Remap patch scores to point scores.

    Args:
        patch_scores: (num_patches,) numpy array of patch scores
        patch_indices: (num_patches, patch_size) torch tensor of indices
        num_points: Total number of points

    Returns:
        point_scores: (num_points,) numpy array
    """
    point_scores = np.zeros(num_points, dtype=np.float32)
    point_counts = np.zeros(num_points, dtype=np.int32)

    # Convert indices to CPU numpy
    patch_indices_cpu = patch_indices.cpu().numpy()

    # Assign each patch score to all its points
    for i, indices in enumerate(patch_indices_cpu):
        point_scores[indices] += patch_scores[i]
        point_counts[indices] += 1

    # Average scores for points in multiple patches
    mask = point_counts > 0
    point_scores[mask] = point_scores[mask] / point_counts[mask]

    # For uncovered points (rare), use nearest neighbor from covered points
    uncovered_mask = point_counts == 0
    if uncovered_mask.any():
        print(f"Warning: {uncovered_mask.sum()} points not covered by any patch")

    return point_scores


def visualize_scores(points, scores, title="Anomaly Scores", gt_labels=None, camera_position=None):
    """Visualize anomaly scores on point cloud."""
    visualizer = PyVistaAnomalyVisualizer(
        colormap='jet',
        interactive=True,
        point_size=9.0,
        show_scalar_bar=True,
        window_size=(1200, 900),
        show_boundary=False,
        show_reference_frame=False,
        camera_position=camera_position
    )
    visualizer.visualize(points, scores, sample_name=title, gt_labels=gt_labels)


def main(args):
    """Main pipeline."""
    # Load config
    config = EasyDict()
    cfg_from_yaml_file(args.config, config)

    category = args.category
    dataset_path = config.dataset_path
    dataset_type = config.dataset

    seed_all(args.seed)

    # Load training samples
    train_paths = load_train_samples(dataset_path, category, dataset_type)

    # Estimate voxel size
    first_points = load_pointcloud(train_paths[0], dataset_type)
    diameter = np.linalg.norm(first_points.max(axis=0) - first_points.min(axis=0))
    voxel_size = diameter * 0.05
    print(f"Voxel size: {voxel_size:.6f}")

    # Build memory bank
    memory_bank = build_memory_bank(train_paths, dataset_type, voxel_size,
                                    num_patches=args.num_patches, patch_size=args.patch_size)

    # Find test samples
    test_samples = find_test_samples(dataset_path, category, dataset_type, args.test_file)

    print(f"\nProcessing {len(test_samples)} test samples...")

    # Initialize lists for metrics computation
    image_preds = []  # Image-level predictions (max score per sample)
    image_labels = []  # Image-level labels (0=normal, 1=anomalous)
    pixel_preds = []  # Pixel-level predictions (all point scores)
    pixel_labels = []  # Pixel-level labels (all point ground truth)

    # Process each test sample
    for subdir, test_file, test_file_path in test_samples:
        print(f"\n{'='*80}")
        print(f"Processing: {subdir}/{test_file}")
        print('='*80)

        # Load test pointcloud
        test_data = load_test_pointcloud(test_file_path, dataset_type, subdir, is_evaluate=True)
        points = test_data['points_normalized']
        gt_mask = test_data['gt_mask']

        # Sample 150k points
        if len(points) > 150000:
            sample_indices = fpsample.bucket_fps_kdline_sampling(points, 150000, h=5)
            points_sampled = points[sample_indices]
        else:
            sample_indices = np.arange(len(points))
            points_sampled = points

        print(f"Sampled {len(points_sampled)} points")

        # Convert to GPU
        points_gpu = torch.from_numpy(points_sampled).float().cuda()

        # Compute FPFH
        print(f"Computing FPFH features...")
        fpfh_gpu = compute_fpfh_features(points_gpu, voxel_size)

        # Extract patches and average FPFH
        print(f"Extracting {args.num_patches} patches of {args.patch_size} points...")
        patch_fpfh_avg, patch_centers, patch_indices = extract_patches_fps_knn(
            points_gpu, fpfh_gpu, args.num_patches, args.patch_size
        )

        # Compute anomaly scores
        print(f"Computing anomaly scores (k={args.k_nn})...")
        patch_scores_gpu = compute_anomaly_scores(patch_fpfh_avg, memory_bank['fpfh'], k=args.k_nn)
        patch_scores = patch_scores_gpu.cpu().numpy()

        # Remap patch scores to sampled points
        print(f"Remapping scores to points...")
        point_scores_sampled = remap_patch_scores_to_points(patch_scores, patch_indices, len(points_sampled))

        # Remap to full point cloud if needed
        if len(sample_indices) < len(points):
            point_scores_full = np.zeros(len(points), dtype=np.float32)
            point_scores_full[sample_indices] = point_scores_sampled

            # NN for unsampled points
            unsampled_mask = np.ones(len(points), dtype=bool)
            unsampled_mask[sample_indices] = False
            unsampled_indices = np.where(unsampled_mask)[0]

            if len(unsampled_indices) > 0:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(points[sample_indices])
                _, nn_indices = nn.kneighbors(points[unsampled_indices])
                point_scores_full[unsampled_indices] = point_scores_sampled[nn_indices.flatten()]
        else:
            point_scores_full = point_scores_sampled

        # Print statistics
        print(f"\nAnomaly score statistics:")
        print(f"  Mean: {point_scores_full.mean():.6f}")
        print(f"  Std:  {point_scores_full.std():.6f}")
        print(f"  Min:  {point_scores_full.min():.6f}")
        print(f"  Max:  {point_scores_full.max():.6f}")

        is_anomalous = 'good' not in test_file.lower() and 'positive' not in test_file.lower()
        print(f"  Ground truth: {'ANOMALOUS' if is_anomalous else 'NORMAL'}")

        # Accumulate for metrics computation
        # Image-level: Use max score as image-level prediction
        image_preds.append(point_scores_full.max())
        image_labels.append(1 if is_anomalous else 0)

        # Point-level: Accumulate all point scores and ground truth labels
        pixel_preds.append(point_scores_full)
        pixel_labels.append(gt_mask)

        # Visualize
        if not args.no_viz:
            sample_name = f"{category} - {subdir}/{test_file} (FPFH Patch k-NN, k={args.k_nn})"
            visualize_scores(points, point_scores_full, sample_name, gt_labels=gt_mask, camera_position=args.cam)

    # Compute and print metrics
    print(f"\n{'='*80}")
    print("EVALUATION METRICS")
    print('='*80)

    # Convert lists to arrays
    image_preds = np.array(image_preds)
    image_labels = np.array(image_labels)
    pixel_preds = np.concatenate(pixel_preds)
    pixel_labels = np.concatenate(pixel_labels)

    # Compute metrics
    compute_metrics(
        image_preds=image_preds,
        image_labels=image_labels,
        pixel_preds=pixel_preds,
        pixel_labels=pixel_labels,
        dataset_type=dataset_type,
        category=category,
        fusion_p=args.k_nn  # Use k_nn as a proxy for method parameter
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Dataset config file')
    parser.add_argument('--category', type=str, required=True, help='Category name')
    parser.add_argument('--test_file', type=str, default=None, help='Specific test file')
    parser.add_argument('--k_nn', type=int, default=1, help='Number of nearest neighbors')
    parser.add_argument('--num_patches', type=int, default=2048, help='Number of patches')
    parser.add_argument('--patch_size', type=int, default=128, help='Points per patch')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    parser.add_argument('--cam', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'), help='Camera position (x y z)')

    args = parser.parse_args()
    main(args)