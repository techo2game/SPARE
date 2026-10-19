"""
Seed-based deformation methods with mesh-aware sculpting.

Clean implementation - removed old mask-based component deformations (saved as backup).
"""
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


# =============================================================================
# Utility functions
# =============================================================================

def compute_point_normals(pointcloud, k_neighbors=30):
    """
    Compute per-point surface normals for the entire point cloud.

    Normals are oriented consistently OUTWARD from the object's centroid.
    This ensures coherent deformation (all bulge or all dent, not mixed).

    Args:
        pointcloud: (N, 3) numpy array
        k_neighbors: Number of neighbors for normal estimation (default: 30)

    Returns:
        normals: (N, 3) numpy array of normalized normal vectors (oriented outward)
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))

    normals = np.asarray(pcd.normals)

    # Orient normals consistently OUTWARD (away from centroid)
    # This is critical for coherent sculpting deformation
    centroid = np.mean(pointcloud, axis=0)
    vectors_from_centroid = pointcloud - centroid  # (N, 3)

    # Check if normal points outward (dot product > 0) or inward (dot product < 0)
    dot_products = np.sum(normals * vectors_from_centroid, axis=1)  # (N,)

    # Flip normals that point inward (dot product < 0)
    normals[dot_products < 0] *= -1

    return normals


def apply_distance_based_ground_truth(pointcloud, deformed_points, distance_threshold=0.05):
    """
    Compute distance-based ground truth mask.

    Args:
        pointcloud: (N, 3) numpy array of original point cloud
        deformed_points: (N, 3) numpy array of deformed point cloud
        distance_threshold: float, minimum distance from original position to be considered anomalous

    Returns:
        binary_mask: (N,) binary mask (0=normal, 1=anomaly) based on distance from original
    """
    # Compute Euclidean distance from original positions
    distances = np.linalg.norm(deformed_points - pointcloud, axis=1)

    # Points that moved more than the distance threshold are considered anomalous
    binary_mask = (distances >= distance_threshold).astype(np.float32)

    return binary_mask


# =============================================================================
# Seed-based deformation (mesh-aware sculpting)
# =============================================================================

def apply_seed_based_deformation(
    pointcloud: np.ndarray,
    seed_data: dict,
    influence_radius: float = 0.15,
    amplitude: float = 0.1,
    distance_threshold: float = 0.05,
    falloff_type: str = 'gaussian',
    smoothing_iterations: int = 3,
    k_neighbors: int = 20,
    direction: float = None,
    **kwargs
):
    """
    Apply mesh-aware sculpting deformation from seeds.

    Proper sculpting that maintains surface connectivity:
    1. Compute distance from seeds
    2. Apply deformation with smooth falloff
    3. Use mesh constraints to maintain continuity (Laplacian smoothing)

    Args:
        pointcloud: (N, 3) array of original points
        seed_data: Seed information dict (contains seed_points, seed_indices)
        influence_radius: Maximum deformation radius
        amplitude: Deformation strength at seed
        distance_threshold: GT distance threshold (points moving > this = anomaly)
        falloff_type: 'gaussian', 'linear', or 'cosine'
        smoothing_iterations: Number of Laplacian smoothing iterations
        k_neighbors: Neighbors for mesh connectivity
        direction: +1 for bulge, -1 for dent, None for random

    Returns:
        deformed_points: (N, 3) deformed point cloud
        binary_mask: (N,) ground truth mask
    """
    N = len(pointcloud)
    seed_points = seed_data['seed_points']
    seed_indices = seed_data['seed_indices']

    # Choose deformation direction (bulge or dent)
    if direction is None:
        direction = random.choice([1.0, -1.0])
    else:
        direction = float(direction)

    # Compute per-point normals
    normals = compute_point_normals(pointcloud, k_neighbors=30)

    # Build KNN graph for mesh connectivity
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(pointcloud)
    distances_graph, indices_graph = nbrs.kneighbors(pointcloud)

    # Compute distance from each point to nearest seed
    dist_to_seeds = cdist(pointcloud, seed_points)  # (N, S)
    min_dist_to_seed = np.min(dist_to_seeds, axis=1)  # (N,)

    # Compute falloff weights based on distance
    if falloff_type == 'gaussian':
        # Gaussian falloff: exp(-dist^2 / (2 * (radius/3)^2))
        sigma = influence_radius / 3.0
        falloff_weights = np.exp(-min_dist_to_seed**2 / (2 * sigma**2))
    elif falloff_type == 'linear':
        # Linear falloff: 1 - dist/radius
        falloff_weights = np.maximum(0, 1 - min_dist_to_seed / influence_radius)
    elif falloff_type == 'cosine':
        # Cosine falloff: 0.5 * (1 + cos(pi * dist / radius))
        falloff_weights = np.where(
            min_dist_to_seed <= influence_radius,
            0.5 * (1 + np.cos(np.pi * min_dist_to_seed / influence_radius)),
            0.0
        )
    else:
        raise ValueError(f"Unknown falloff_type: {falloff_type}. Use 'gaussian', 'linear', or 'cosine'")

    # Clip weights to influence radius
    falloff_weights[min_dist_to_seed > influence_radius] = 0.0

    # Initial displacement: normal direction with falloff
    initial_displacements = normals * amplitude * direction * falloff_weights[:, np.newaxis]
    deformed_points = pointcloud + initial_displacements

    # Apply mesh-aware smoothing to maintain surface continuity
    affected_mask = falloff_weights > 0.01
    affected_indices = np.where(affected_mask)[0]

    if len(affected_indices) > 0:
        # Iterative smoothing of deformed positions
        for iteration in range(smoothing_iterations):
            smoothed = deformed_points.copy()

            for idx in affected_indices:
                # Get neighbors of this point
                neighbor_indices = indices_graph[idx]
                neighbor_distances = distances_graph[idx]

                # Weighted average based on inverse distance in ORIGINAL mesh
                weights = 1.0 / (neighbor_distances + 1e-8)
                weights = weights / weights.sum()

                # Smooth deformed position with neighbors
                smoothed_pos = np.sum(deformed_points[neighbor_indices] * weights[:, np.newaxis], axis=0)

                # Blend: strong smoothing for high falloff, less for low falloff
                blend_factor = falloff_weights[idx]
                smoothed[idx] = blend_factor * smoothed_pos + (1 - blend_factor) * deformed_points[idx]

            deformed_points = smoothed

    # Compute distance-based ground truth
    binary_mask = apply_distance_based_ground_truth(pointcloud, deformed_points, distance_threshold)

    return deformed_points, binary_mask
