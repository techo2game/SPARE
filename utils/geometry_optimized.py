"""
Optimized geometry-based feature computation

"""
import torch


def compute_knn_once(patch_center_batch, reference_patch_centers, k_centers=10):
    """
    Compute k-nearest reference centers.

    Args:
        patch_center_batch: (P, 3) test patch centers
        reference_patch_centers: (R, 3) reference centers
        k_centers: Number of nearest centers

    Returns:
        nearest_indices: (P, k) indices of k nearest centers for each patch
        center_dists: (P, R) full distance matrix (for reuse if needed)
    """
    k = min(k_centers, reference_patch_centers.shape[0])
    center_dists = torch.cdist(patch_center_batch.float(), reference_patch_centers.float())
    _, nearest_indices = torch.topk(center_dists, k, dim=1, largest=False)
    return nearest_indices, center_dists


def compute_patch_normals_batch(patch_points):
    """
    Vectorized PCA-based normal computation.

    Args:
        patch_points: (P, patch_size, 3)

    Returns:
        normals: (P, 3)
        centers: (P, 3)
    """
    P, patch_size, _ = patch_points.shape

    # Center patches
    patch_centers = patch_points.mean(dim=1, keepdim=True)  # (P, 1, 3)
    centered = patch_points - patch_centers  # (P, patch_size, 3)

    # Covariance: (P, 3, 3)
    cov = torch.bmm(centered.transpose(1, 2), centered) / patch_size

    # SVD (batch)
    try:
        U, S, V = torch.svd(cov)
        normals = U[:, :, 2]  # (P, 3) 
    except:
        v1 = centered[:, 0, :]
        v2 = centered[:, patch_size // 2, :]
        normals = torch.cross(v1, v2, dim=1)

    # Normalize
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-6)

    return normals, patch_centers.squeeze(1)


def compute_memory_bank_statistics_vectorized(
    patch_points_batch,       # (P, patch_size, 3)
    patch_center_batch,       # (P, 3)
    reference_means,          # (R, N, 3)
    reference_normals,        # (R, N, 3)
    nearest_indices,          # (P, k) 
):
    """

    Args:
        patch_points_batch: (P, patch_size, 3)
        patch_center_batch: (P, 3)
        reference_means: (R, N, 3)
        reference_normals: (R, N, 3)
        nearest_indices: (P, k) precomputed k-nearest centers

    Returns:
        features: (P, 9) [center_xyz, mean_stats(3), normal_stats(3)]
    """
    P = patch_points_batch.shape[0]
    k = nearest_indices.shape[1]
    N = reference_means.shape[1]
    device = patch_points_batch.device

    # Compute test normals/means
    test_normals, test_means = compute_patch_normals_batch(patch_points_batch)  # (P, 3), (P, 3)

    # Gather reference data: (P, k, N, 3)
    nearest_ref_means = reference_means[nearest_indices]  # (P, k, N, 3)
    nearest_ref_normals = reference_normals[nearest_indices]  # (P, k, N, 3)

    # Flatten to (P, k*N, 3)
    nearest_ref_means_flat = nearest_ref_means.reshape(P, k * N, 3)
    nearest_ref_normals_flat = nearest_ref_normals.reshape(P, k * N, 3)

    # Compute distances
    # test_means: (P, 3) → (P, 1, 3)
    mean_dists = torch.norm(
        nearest_ref_means_flat - test_means.unsqueeze(1), dim=2
    )  # (P, k*N)

    # Compute normal angles
    cos_angles = torch.abs(
        (nearest_ref_normals_flat * test_normals.unsqueeze(1)).sum(dim=2)
    )  # (P, k*N)
    normal_diffs = 1.0 - cos_angles  # (P, k*N)

    # Statistics along k*N dimension
    min_mean_dist = mean_dists.min(dim=1)[0]  # (P,)
    mean_of_mean_dists = mean_dists.mean(dim=1)  # (P,)
    std_mean_dist = mean_dists.std(dim=1)  # (P,)

    min_normal_diff = normal_diffs.min(dim=1)[0]  # (P,)
    mean_of_normal_diffs = normal_diffs.mean(dim=1)  # (P,)
    std_normal_diff = normal_diffs.std(dim=1)  # (P,)

    # Stack features (P, 9)
    features = torch.stack([
        patch_center_batch[:, 0],  # x
        patch_center_batch[:, 1],  # y
        patch_center_batch[:, 2],  # z
        min_mean_dist,
        mean_of_mean_dists,
        std_mean_dist,
        min_normal_diff,
        mean_of_normal_diffs,
        std_normal_diff
    ], dim=1)

    return features


def compute_geometry_scores_learned_fusion_vectorized(
    patch_points_batch,       # (P, patch_size, 3)
    reference_means,          # (R, N, 3)
    reference_normals,        # (R, N, 3)
    fusion_weights,           # (P, 2)
    nearest_indices,          # (P, k)
):
    """
    Vectorized geometry score computation 

    Args:
        patch_points_batch: (P, patch_size, 3)
        reference_means: (R, N, 3)
        reference_normals: (R, N, 3)
        fusion_weights: (P, 2) [w_mean, w_normal]
        nearest_indices: (P, k) precomputed k-nearest centers

    Returns:
        scores: (P,) fused geometry scores
    """
    P = patch_points_batch.shape[0]
    k = nearest_indices.shape[1]
    N = reference_means.shape[1]

    # Compute test normals/means
    test_normals, test_means = compute_patch_normals_batch(patch_points_batch)  # (P, 3)

    # Gather reference data: (P, k, N, 3)
    nearest_ref_means = reference_means[nearest_indices]
    nearest_ref_normals = reference_normals[nearest_indices]

    # Flatten to (P, k*N, 3)
    nearest_ref_means_flat = nearest_ref_means.reshape(P, k * N, 3)
    nearest_ref_normals_flat = nearest_ref_normals.reshape(P, k * N, 3)

    # Compute mean distances
    mean_dists = torch.norm(
        nearest_ref_means_flat - test_means.unsqueeze(1), dim=2
    )  # (P, k*N)

    # Compute normal angles
    cos_angles = torch.abs(
        (nearest_ref_normals_flat * test_normals.unsqueeze(1)).sum(dim=2)
    )  # (P, k*N)
    normal_diffs = 1.0 - cos_angles  # (P, k*N)

    # Fuse BEFORE min 
    w_mean = fusion_weights[:, 0].unsqueeze(1)  # (P, 1)
    w_normal = fusion_weights[:, 1].unsqueeze(1)  # (P, 1)
    combined = w_mean * mean_dists + w_normal * normal_diffs  # (P, k*N)

    # Take minimum
    scores = combined.min(dim=1)[0]  # (P,)

    return scores


def compute_fpfh_scores_vectorized(
    patch_fpfh_batch,         # (P, 33)
    reference_patch_fpfh,     # (R, N, 33)
    nearest_indices,          # (P, k) 
):
    """
    Vectorized FPFH score computation.
    Args:
        patch_fpfh_batch: (P, 33)
        reference_patch_fpfh: (R, N, 33)
        nearest_indices: (P, k) precomputed k-nearest centers

    Returns:
        min_dists: (P,) minimum FPFH distances
        best_diffs: (P, 33) FPFH difference to nearest match
    """
    P = patch_fpfh_batch.shape[0]
    k = nearest_indices.shape[1]
    N = reference_patch_fpfh.shape[1]
    device = patch_fpfh_batch.device

    # Gather reference FPFH: (P, k, N, 33)
    nearest_ref_fpfh = reference_patch_fpfh[nearest_indices]

    # Flatten to (P, k*N, 33)
    nearest_ref_fpfh_flat = nearest_ref_fpfh.reshape(P, k * N, 33)

    # Compute differences
    # patch_fpfh_batch: (P, 33) → (P, 1, 33)
    diffs = patch_fpfh_batch.unsqueeze(1) - nearest_ref_fpfh_flat  # (P, k*N, 33)

    # Compute L2 distances
    dists = torch.norm(diffs, dim=2)  # (P, k*N)

    # Find minimum
    min_dists, min_indices = dists.min(dim=1)  # (P,), (P,)

    # Gather best differences
    batch_indices = torch.arange(P, device=device)
    best_diffs = diffs[batch_indices, min_indices]  # (P, 33)

    return min_dists, best_diffs