"""
Geometry-based feature computation for 3D point cloud patches.
"""
import torch


def compute_memory_bank_statistics(
    patch_points_batch,       # (P, patch_size, 3) GPU tensor
    patch_center_batch,       # (P, 3) GPU tensor
    reference_patch_centers,  # (R, 3) GPU tensor
    reference_means,          # (R, N, 3) GPU tensor
    reference_normals,        # (R, N, 3) GPU tensor
    k_centers=10
):
    """
    Compute statistics about local memory bank consistency for fusion weight prediction.

    The idea: Fusion weights should depend on how reliable each metric is in this
    spatial region, not on the test patch features themselves.

    Features computed:
    - patch_center: (3D) spatial location
    - Mean distance stats: min, mean, std of distances to k nearest training means
    - Normal angle stats: min, mean, std of angles to k nearest training normals

    High variance → metric is unreliable in this region → use other metric more
    Low variance → metric is consistent in this region → trust it more

    Args:
        patch_points_batch: (P, patch_size, 3) test patch point clouds
        patch_center_batch: (P, 3) test patch centers
        reference_patch_centers: (R, 3) reference centers
        reference_means: (R, N, 3) PRECOMPUTED reference patch means
        reference_normals: (R, N, 3) PRECOMPUTED reference patch normals
        k_centers: Number of nearby centers to check

    Returns:
        reliability_features: (P, 9) features for fusion weight prediction
            [centroid_x, centroid_y, centroid_z,
             min_mean_dist, mean_mean_dist, std_mean_dist,
             min_normal_diff, mean_normal_diff, std_normal_diff]
    """
    P = patch_points_batch.shape[0]
    R, N, _ = reference_means.shape
    device = patch_points_batch.device

    # Compute test patch normals and means
    test_normals, test_means = compute_patch_normals(patch_points_batch)

    # Find k nearest centers in 3D space
    k = min(k_centers, R)
    center_dists = torch.cdist(patch_center_batch.float(), reference_patch_centers.float())
    _, nearest_indices = torch.topk(center_dists, k, dim=1, largest=False)

    # Compute statistics for each test patch
    feature_list = []

    for p in range(P):
        # Get reference patches from k nearest centers
        nearest_center_indices = nearest_indices[p]
        nearest_ref_means = reference_means[nearest_center_indices]  # (k, N, 3)
        nearest_ref_normals = reference_normals[nearest_center_indices]  # (k, N, 3)

        # Flatten to (k*N, 3)
        nearest_ref_means_flat = nearest_ref_means.reshape(k * N, 3)
        nearest_ref_normals_flat = nearest_ref_normals.reshape(k * N, 3)

        # Compute mean distances (L2 distance between centroids)
        mean_dists = torch.norm(nearest_ref_means_flat - test_means[p].unsqueeze(0), dim=1)

        # Compute normal angles (1 - |cos(angle)|)
        cos_angles = torch.abs(torch.sum(nearest_ref_normals_flat * test_normals[p].unsqueeze(0), dim=1))
        normal_diffs = 1.0 - cos_angles

        # Statistics for mean distances
        min_mean_dist = torch.min(mean_dists)
        mean_of_mean_dists = torch.mean(mean_dists)
        std_mean_dist = torch.std(mean_dists)

        # Statistics for normal angles
        min_normal_diff = torch.min(normal_diffs)
        mean_of_normal_diffs = torch.mean(normal_diffs)
        std_normal_diff = torch.std(normal_diffs)

        # Combine features: centroid + mean_stats + normal_stats
        features = torch.cat([
            patch_center_batch[p],  # (3,) spatial location
            min_mean_dist.unsqueeze(0),
            mean_of_mean_dists.unsqueeze(0),
            std_mean_dist.unsqueeze(0),
            min_normal_diff.unsqueeze(0),
            mean_of_normal_diffs.unsqueeze(0),
            std_normal_diff.unsqueeze(0)
        ])  # (9,)

        feature_list.append(features)

        # Free memory
        del nearest_ref_means, nearest_ref_normals, mean_dists, normal_diffs

    reliability_features = torch.stack(feature_list)  # (P, 9)
    return reliability_features


def compute_patch_normals(patch_points):
    """
    Compute surface normal for each patch using PCA (smallest eigenvector).

    Args:
        patch_points: (num_patches, patch_size, 3) patch points on GPU or CPU

    Returns:
        normals: (num_patches, 3) unit normals
        centers: (num_patches, 3) patch centers (mean positions)
    """
    num_patches, patch_size, _ = patch_points.shape

    # Center each patch
    patch_centers = patch_points.mean(dim=1, keepdim=True)  # (num_patches, 1, 3)
    centered_patches = patch_points - patch_centers  # (num_patches, patch_size, 3)

    # Compute covariance matrix: (num_patches, 3, 3)
    cov = torch.bmm(centered_patches.transpose(1, 2), centered_patches) / patch_size

    # Compute eigenvalues/eigenvectors using SVD
    try:
        U, S, V = torch.svd(cov)
        # Normal is eigenvector with smallest eigenvalue (3rd column)
        normals = U[:, :, 2]  # (num_patches, 3)
    except:
        # Fallback: cross product of two vectors
        v1 = centered_patches[:, 0, :]
        v2 = centered_patches[:, patch_size//2, :]
        normals = torch.cross(v1, v2, dim=1)

    # Normalize
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-6)

    return normals, patch_centers.squeeze(1)


def compute_geometry_scores_learned_fusion(
    patch_points_batch,       # (P, patch_size, 3) GPU tensor
    patch_center_batch,       # (P, 3) GPU tensor
    reference_patch_centers,  # (R, 3) GPU tensor
    reference_means,          # (R, N, 3) GPU tensor
    reference_normals,        # (R, N, 3) GPU tensor
    fusion_weights,           # (P, 2) GPU tensor
    k_centers=10
):
    """
    Compute geometry-based anomaly scores with LEARNED FUSION BEFORE MIN.

    Key idea: Fuse mean_dist and normal_diff with learned weights BEFORE taking min.
    This ensures the min finds the best SINGLE coherent surface point.

    Args:
        patch_points_batch: (P, patch_size, 3) test patch point clouds
        patch_center_batch: (P, 3) test patch centers
        reference_patch_centers: (R, 3) reference centers
        reference_means: (R, N, 3) PRECOMPUTED reference patch means
        reference_normals: (R, N, 3) PRECOMPUTED reference patch normals
        fusion_weights: (P, 2) learned weights [w_mean, w_normal]
        k_centers: Number of nearby centers to check

    Returns:
        scores: (P,) Fused scores (min of weighted combination)
    """
    P = patch_points_batch.shape[0]
    R, N, _ = reference_means.shape
    device = patch_points_batch.device

    # Compute test patch normals and means together
    test_normals, test_means = compute_patch_normals(patch_points_batch)

    # Find k nearest centers in 3D space
    k = min(k_centers, R)
    center_dists = torch.cdist(patch_center_batch.float(), reference_patch_centers.float())
    _, nearest_indices = torch.topk(center_dists, k, dim=1, largest=False)

    # For each test patch, compute fused score
    score_list = []

    for p in range(P):
        # Get means and normals from k nearest centers
        nearest_center_indices = nearest_indices[p]
        nearest_ref_means = reference_means[nearest_center_indices]  # (k, N, 3)
        nearest_ref_normals = reference_normals[nearest_center_indices]  # (k, N, 3)

        # Flatten to (k*N, 3)
        nearest_ref_means_flat = nearest_ref_means.reshape(k * N, 3)
        nearest_ref_normals_flat = nearest_ref_normals.reshape(k * N, 3)

        # Compute mean distance (L2 distance between centroids)
        mean_dists = torch.norm(nearest_ref_means_flat - test_means[p].unsqueeze(0), dim=1)

        # Compute normal angle (1 - |cos(angle)|)
        cos_angles = torch.abs(torch.sum(nearest_ref_normals_flat * test_normals[p].unsqueeze(0), dim=1))
        normal_diffs = 1.0 - cos_angles

        # FUSE BEFORE MIN with learned weights (preserves surface coherence!)
        w_mean = fusion_weights[p, 0]
        w_normal = fusion_weights[p, 1]
        combined_scores = w_mean * mean_dists + w_normal * normal_diffs

        # Take minimum of fused scores
        min_score = torch.min(combined_scores)
        score_list.append(min_score)

        # Free memory
        del nearest_ref_means, nearest_ref_normals, mean_dists, normal_diffs, combined_scores

    scores = torch.stack(score_list)
    return scores


def compute_geometry_scores(
    patch_points_batch,       # (P, patch_size, 3) GPU tensor
    patch_center_batch,       # (P, 3) GPU tensor
    reference_patch_centers,  # (R, 3) GPU tensor
    reference_means,          # (R, N, 3) GPU tensor
    reference_normals,        # (R, N, 3) GPU tensor
    k_centers=10
):
    """
    OLD FUNCTION: Compute geometry scores SEPARATELY (for caching/evaluation).

    This is kept for backward compatibility with precomputed test features.
    For training with learned fusion, use compute_geometry_scores_learned_fusion instead.
    """
    P = patch_points_batch.shape[0]
    R, N, _ = reference_means.shape

    test_normals, test_means = compute_patch_normals(patch_points_batch)

    k = min(k_centers, R)
    center_dists = torch.cdist(patch_center_batch.float(), reference_patch_centers.float())
    _, nearest_indices = torch.topk(center_dists, k, dim=1, largest=False)

    score_mean_list = []
    score_normal_list = []

    for p in range(P):
        nearest_center_indices = nearest_indices[p]
        nearest_ref_means = reference_means[nearest_center_indices]
        nearest_ref_normals = reference_normals[nearest_center_indices]
        nearest_ref_means_flat = nearest_ref_means.reshape(k * N, 3)
        nearest_ref_normals_flat = nearest_ref_normals.reshape(k * N, 3)

        mean_dists = torch.norm(nearest_ref_means_flat - test_means[p].unsqueeze(0), dim=1)
        cos_angles = torch.abs(torch.sum(nearest_ref_normals_flat * test_normals[p].unsqueeze(0), dim=1))
        normal_diffs = 1.0 - cos_angles

        min_mean_dist = torch.min(mean_dists)
        min_normal_diff = torch.min(normal_diffs)

        score_mean_list.append(min_mean_dist)
        score_normal_list.append(min_normal_diff)

        del nearest_ref_means, nearest_ref_normals, mean_dists, normal_diffs

    score_mean_dist = torch.stack(score_mean_list)
    score_normal_diff = torch.stack(score_normal_list)
    return score_mean_dist, score_normal_diff