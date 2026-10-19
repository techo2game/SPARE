"""
Patch extraction and sampling utilities for 3D point clouds.
"""
import torch
import numpy as np
try:
    from torch_cluster import knn
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False
from pykeops.torch import LazyTensor
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from cuda_ops.fps_kdtree.fps_kdtree_quickfps_gpu import fps_kdtree_quickfps


def extract_patches_gpu_fps(test_aligned, test_fpfh, num_patches, patch_size):
    """
    Extract patches efficiently on GPU using FPS for center selection and sklearn KNN.

    Args:
        test_aligned: Aligned test point cloud (N, 3) torch tensor on GPU
        test_fpfh: Test FPFH features (N, 33) torch tensor on GPU
        num_patches: Number of patches to extract
        patch_size: Points per patch

    Returns:
        tuple: (patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, patch_indices)
    """
    device = test_aligned.device
    N = test_aligned.shape[0]

    # Use QuickFPS (bucket-based FPS with KDTree pruning)
    center_indices = fps_kdtree_quickfps(test_aligned, num_patches, bucket_size=2000)

    # Get patch centers
    patch_centers_gpu = test_aligned[center_indices]  # (num_patches, 3)

    # KNN using sklearn (CPU-based, but reliable and memory-efficient)
    points_cpu = test_aligned.detach().cpu().numpy()  # (N, 3)
    centers_cpu = patch_centers_gpu.detach().cpu().numpy()  # (num_patches, 3)

    # Build KNN index
    nbrs = NearestNeighbors(n_neighbors=patch_size, algorithm='auto', n_jobs=-1)
    nbrs.fit(points_cpu)

    # Find k-nearest neighbors for each center
    distances, indices = nbrs.kneighbors(centers_cpu)  # (num_patches, patch_size)

    # Convert back to torch tensors on GPU
    patch_indices_tensor = torch.from_numpy(indices).long().to(device)  # (num_patches, patch_size)

    # Gather patch points and FPFH
    patch_points_gpu = test_aligned[patch_indices_tensor]  # (num_patches, patch_size, 3)
    patch_fpfh_gpu = test_fpfh[patch_indices_tensor].mean(dim=1)  # (num_patches, 33)

    return patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, patch_indices_tensor


def extract_patches_gpu_random(test_aligned, test_fpfh, num_patches, patch_size):
    """
    Extract patches efficiently on GPU using RANDOM center selection and fast KNN.

    Args:
        test_aligned: Aligned test point cloud (N, 3) torch tensor on GPU
        test_fpfh: Test FPFH features (N, 33) torch tensor on GPU
        num_patches: Number of patches to extract
        patch_size: Points per patch

    Returns:
        tuple: (patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, patch_indices)
    """
    device = test_aligned.device
    N = test_aligned.shape[0]

    # Random center selection
    center_indices = torch.randperm(N, device=device)[:num_patches]
    patch_centers_gpu = test_aligned[center_indices]  # (num_patches, 3)

    # Fast KNN using torch_cluster
    batch_x = torch.zeros(N, dtype=torch.long, device=device)
    batch_y = torch.zeros(num_patches, dtype=torch.long, device=device)

    edge_index = knn(test_aligned, patch_centers_gpu, patch_size, batch_x, batch_y)
    patch_indices_tensor = edge_index[0].reshape(num_patches, patch_size)

    # Gather patch points and FPFH
    patch_points_gpu = test_aligned[patch_indices_tensor]  # (num_patches, patch_size, 3)
    patch_fpfh_gpu = test_fpfh[patch_indices_tensor].mean(dim=1)  # (num_patches, 33)

    return patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, patch_indices_tensor


def collate_fn_with_patches(batch, num_patches, patch_size):
    """
    Custom collate function that extracts patches on CPU using KDTree.

    This runs in DataLoader workers in parallel

    Args:
        batch: List of samples from dataset, each with 'points', 'fpfh', 'mask'
        num_patches: Number of patches to extract per sample
        patch_size: Points per patch

    Returns:
        Batched dict with patch_points, patch_fpfh, patch_centers, patch_indices, masks
    """
    batch_patch_points = []
    batch_patch_fpfh = []
    batch_patch_centers = []
    batch_patch_indices = []
    batch_masks = []
    batch_splits = []
    batch_points = []  # Store original points for visualization

    for sample in batch:
        points = sample['points'].numpy()  # (N, 3)
        fpfh = sample['fpfh'].numpy()      # (N, 33)
        mask = sample['mask'].numpy()      # (N,)
        N = points.shape[0]

        # Random center selection
        center_indices = np.random.permutation(N)[:num_patches]
        centers = points[center_indices]  # (num_patches, 3)

        # Fast KDTree KNN on CPU
        tree = cKDTree(points)
        _, patch_indices = tree.query(centers, k=patch_size, workers=1)  # (num_patches, patch_size)

        # Gather patches
        patch_points = points[patch_indices]  # (num_patches, patch_size, 3)
        patch_fpfh = fpfh[patch_indices].mean(axis=1)  # (num_patches, 33)

        # Convert to tensors
        batch_patch_points.append(torch.from_numpy(patch_points))
        batch_patch_fpfh.append(torch.from_numpy(patch_fpfh))
        batch_patch_centers.append(torch.from_numpy(centers))
        batch_patch_indices.append(torch.from_numpy(patch_indices))
        batch_masks.append(torch.from_numpy(mask))
        batch_splits.append(sample['split'])
        batch_points.append(torch.from_numpy(points))  # Store original points

    # Stack into batches
    return {
        'patch_points': torch.stack(batch_patch_points),      # (B, num_patches, patch_size, 3)
        'patch_fpfh': torch.stack(batch_patch_fpfh),          # (B, num_patches, 33)
        'patch_centers': torch.stack(batch_patch_centers),    # (B, num_patches, 3)
        'patch_indices': batch_patch_indices,                  # List of (num_patches, patch_size) tensors
        'masks': batch_masks,                                  # List of (N,) tensors
        'splits': batch_splits,
        'points': batch_points                                 # List of (N, 3) tensors - original points
    }