"""
Patch feature computation for FPFH + Geometry fusion.
"""
import torch
from .geometry import compute_geometry_scores


def compute_fpfh_scores(
    patch_fpfh_batch,         # (P, 33) GPU tensor
    patch_center_batch,       # (P, 3) GPU tensor
    reference_patch_centers,  # (R, 3) GPU tensor
    reference_patch_fpfh,     # (R, N, 33) GPU tensor
    k_centers=10
):
    """
    Compute FPFH-based anomaly scores + difference vectors.

    Args:
        patch_fpfh_batch: (P, 33) test patch FPFH features
        patch_center_batch: (P, 3) test patch centers
        reference_patch_centers: (R, 3) reference centers
        reference_patch_fpfh: (R, N, 33) reference FPFH organized by centers
        k_centers: Number of nearby centers to check

    Returns:
        score_fpfh: (P,) FPFH branch scores
        fpfh_diff: (P, 33) FPFH difference to nearest reference
    """
    P = patch_fpfh_batch.shape[0]
    R, N, _ = reference_patch_fpfh.shape
    device = patch_fpfh_batch.device

    # Find k nearest centers in 3D space
    k = min(k_centers, R)
    center_dists = torch.cdist(patch_center_batch.float(), reference_patch_centers.float())
    _, nearest_indices = torch.topk(center_dists, k, dim=1, largest=False)

    # Process k centers one at a time 
    min_dists_fpfh = torch.full((P,), float('inf'), device=device)
    best_fpfh_diff = torch.zeros((P, 33), device=device)

    for ki in range(k):
        center_idx = nearest_indices[:, ki]
        center_features = reference_patch_fpfh[center_idx]  # (P, N, 33)

        # Compute differences
        diffs = patch_fpfh_batch.unsqueeze(1) - center_features
        dists = torch.norm(diffs, dim=2)

        # Find min for this center
        local_min, local_min_idx = dists.min(dim=1)

        # Update global minimum
        for p in range(P):
            if local_min[p] < min_dists_fpfh[p]:
                min_dists_fpfh[p] = local_min[p]
                best_fpfh_diff[p] = diffs[p, local_min_idx[p]]

        del center_features, diffs, dists

    return min_dists_fpfh, best_fpfh_diff


def compute_patch_features_for_network(
    patch_points_batch,       # (P, patch_size, 3) GPU tensor
    patch_center_batch,       # (P, 3) GPU tensor
    reference_patch_centers,  # (R, 3) GPU tensor
):
    """
    Compute geometric patch characteristics.

    Features computed:
    1-3. Patch center position (x, y, z) - meaningful after registration
    4-6. Closest reference center position (x_ref, y_ref, z_ref)
    7. L2 distance to closest reference center

    Args:
        patch_points_batch: (P, patch_size, 3) patch point clouds
        patch_center_batch: (P, 3) patch centers
        reference_patch_centers: (R, 3) reference centers

    Returns:
        patch_features: (P, 7) geometric features [x, y, z, x_ref, y_ref, z_ref, l2_dist]
    """
    P = patch_points_batch.shape[0]

    # Distance to nearest reference center
    center_dists = torch.cdist(patch_center_batch.float(), reference_patch_centers.float())
    min_center_dist, nearest_idx = center_dists.min(dim=1)  # (P,), (P,)

    # Get closest reference center coordinates
    closest_ref_centers = reference_patch_centers[nearest_idx]  # (P, 3)

    # Patch center position (after registration)
    center_x = patch_center_batch[:, 0]
    center_y = patch_center_batch[:, 1]
    center_z = patch_center_batch[:, 2]

    # Closest reference center position
    ref_x = closest_ref_centers[:, 0]
    ref_y = closest_ref_centers[:, 1]
    ref_z = closest_ref_centers[:, 2]

    patch_features = torch.stack([
        center_x,
        center_y,
        center_z,
        ref_x,
        ref_y,
        ref_z,
        min_center_dist,
    ], dim=1)  # (P, 7)

    return patch_features

def compute_patch_features_fpfh_batch(
    patch_points_batch,       # (P, patch_size, 3) GPU tensor
    patch_fpfh_batch,         # (P, 33) GPU tensor
    patch_center_batch,       # (P, 3) GPU tensor
    reference_patch_centers,  # (R, 3) GPU tensor
    reference_means,          # (R, N, 3) GPU tensor
    reference_normals,        # (R, N, 3) GPU tensor
    reference_patch_fpfh,     # (R, N, 33) GPU tensor
    k_centers=10
):
    """
    Compute three separate anomaly scores + rich patch features for the network.

    Returns:
        patch_features: (P, 37) = geometric(4) + fpfh_diff(33)
        score_fpfh: (P,) FPFH branch scores
        score_mean_dist: (P,) Mean distance scores
        score_normal_diff: (P,) Normal angle difference scores
    """
    # Compute geometric features (4D: spatial + position)
    geometric_features = compute_patch_features_for_network(
        patch_points_batch,
        patch_center_batch,
        reference_patch_centers
    )

    # # Compute Geometry scores separately (mean distance and normal angle)
    # score_mean_dist, score_normal_diff = compute_geometry_scores(
    #     patch_points_batch,
    #     patch_center_batch,
    #     reference_patch_centers,
    #     reference_means,
    #     reference_normals,
    #     k_centers=k_centers
    # )

    # Compute FPFH score + difference vector (33D)
    score_fpfh, fpfh_diff = compute_fpfh_scores(
        patch_fpfh_batch,
        patch_center_batch,
        reference_patch_centers,
        reference_patch_fpfh,
        k_centers=k_centers
    )

    # Concatenate geometric features + FPFH difference
    patch_features = torch.cat([
        geometric_features,  # (P, 4)
    ], dim=1)  # (P, 37)

    return patch_features, score_fpfh


def compute_patch_features_fpfh_chamfer_batch(
    patch_points_batch,       # (P, patch_size, 3) GPU tensor
    patch_fpfh_batch,         # (P, 33) GPU tensor
    patch_center_batch,       # (P, 3) GPU tensor
    reference_patch_centers,  # (R, 3) GPU tensor
    reference_means,          # (R, N, 3) GPU tensor
    reference_normals,        # (R, N, 3) GPU tensor
    reference_patch_fpfh,     # (R, N, 33) GPU tensor
    k_centers=10
):
    """
    Compute three separate anomaly scores + rich patch features for the network.

    Returns:
        patch_features: (P, 7) = geometric features [x, y, z, x_ref, y_ref, z_ref, l2_dist]
        score_fpfh: (P,) FPFH branch scores
        score_mean_dist: (P,) Mean distance scores
        score_normal_diff: (P,) Normal angle difference scores
    """
    # Compute geometric features (7D: test center + ref center + distance)
    patch_features = compute_patch_features_for_network(
        patch_points_batch,
        patch_center_batch,
        reference_patch_centers
    )

    # Compute Geometry scores separately (mean distance and normal angle)
    score_mean_dist, score_normal_diff = compute_geometry_scores(
        patch_points_batch,
        patch_center_batch,
        reference_patch_centers,
        reference_means,
        reference_normals,
        k_centers=k_centers
    )

    # Compute FPFH score
    score_fpfh, _ = compute_fpfh_scores(
        patch_fpfh_batch,
        patch_center_batch,
        reference_patch_centers,
        reference_patch_fpfh,
        k_centers=k_centers
    )

    return patch_features, score_fpfh, score_mean_dist, score_normal_diff