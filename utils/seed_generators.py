"""
Seed generators for seed-based synthetic anomaly generation.

Seeds define approximate locations for deformation (lines or points).
Unlike masks, seeds don't specify the full affected region - that's determined
by propagation methods.
"""

import numpy as np
import open3d as o3d
from typing import Dict, Tuple, Callable
from sklearn.neighbors import NearestNeighbors


def generate_line_seed_fps(
    pointcloud: np.ndarray,
    num_seeds: int = 3,
    line_length: int = 50,
    k_neighbors: int = 5,
    use_fps: bool = True,
    seed: int = None,
    **kwargs
) -> Dict:
    """
    Generate line seeds using FPS or random sampling for start points + KNN-based line growth.

    Args:
        pointcloud: (N, 3) array of points
        num_seeds: Number of line seeds to generate
        line_length: Number of points in each line seed (absolute count)
        k_neighbors: Number of neighbors for line growth
        use_fps: If True, use FPS sampling; if False, use random sampling
        seed: Random seed for reproducibility (only used for random sampling)

    Returns:
        SeedData dict with:
            - seed_type: 'line'
            - seed_points: (S, 3) coordinates of line points
            - seed_normals: (S, 3) normals at line points
            - seed_indices: (S,) indices in original point cloud
            - seed_weights: (S,) per-seed importance weights
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(pointcloud)

    print(f"\n[DEBUG Line Seed Generator]")
    print(f"  Point cloud: N={N}")
    print(f"  Target: {num_seeds} lines × {line_length} points/line")

    # Sample starting points for lines
    if use_fps:
        # Use fast FPS to sample starting points (deterministic based on point cloud)
        import fpsample
        start_indices = fpsample.bucket_fps_kdline_sampling(pointcloud, num_seeds, h=5)
        start_indices = np.array(start_indices, dtype=np.int64)
    else:
        # Random sampling (non-deterministic unless seed is set)
        start_indices = np.random.choice(N, size=num_seeds, replace=False)

    # Build KNN tree for line growth
    # Use larger k for KNN graph to allow long lines to grow
    # k_neighbors_graph should be >> line_length to avoid getting stuck
    k_neighbors_graph = max(k_neighbors, min(100, N // 10))
    print(f"  Using k_neighbors={k_neighbors_graph} for line growth")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors_graph, algorithm='kd_tree').fit(pointcloud)

    all_line_indices = []

    for line_idx, start_idx in enumerate(start_indices):
        line_indices = [start_idx]
        current_idx = start_idx
        current_pos = pointcloud[start_idx]

        # Initialize direction as random (will be refined by line growth)
        prev_direction = np.random.randn(3)
        prev_direction /= np.linalg.norm(prev_direction) + 1e-8

        # Grow line by hopping to neighbors with directional continuity
        # Stop when we reach the target number of points
        while len(line_indices) < line_length:
            # Find neighbors
            distances, indices = nbrs.kneighbors([current_pos])
            distances = distances[0]
            indices = indices[0]

            # Exclude already visited points
            candidate_mask = np.array([idx not in line_indices for idx in indices])
            if not candidate_mask.any():
                break

            candidate_indices = indices[candidate_mask]
            candidate_positions = pointcloud[candidate_indices]

            # Compute directions to candidates
            directions = candidate_positions - current_pos
            direction_norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
            directions = directions / direction_norms

            # Score by directional continuity (dot product with previous direction)
            continuity_scores = np.dot(directions, prev_direction)

            # Choose candidate with best continuity
            best_candidate_idx = np.argmax(continuity_scores)
            next_idx = candidate_indices[best_candidate_idx]
            next_pos = pointcloud[next_idx]

            # Update line
            line_indices.append(next_idx)

            # Update for next iteration
            step_length = np.linalg.norm(next_pos - current_pos)
            prev_direction = (next_pos - current_pos) / (step_length + 1e-8)
            current_idx = next_idx
            current_pos = next_pos

        all_line_indices.extend(line_indices)

        actual_line_length = sum(np.linalg.norm(pointcloud[line_indices[i+1]] - pointcloud[line_indices[i]])
                                  for i in range(len(line_indices)-1))
        print(f"  Line {line_idx+1}: {len(line_indices)} points, "
              f"spatial length={actual_line_length:.4f}")

    all_line_indices = list(set(all_line_indices))
    seed_indices = np.array(all_line_indices)
    seed_points = pointcloud[seed_indices]

    print(f"Total seed points after deduplication: {len(seed_indices)} (from {num_seeds} lines)")

    # Compute normals at seed points using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    all_normals = np.asarray(pcd.normals)
    seed_normals = all_normals[seed_indices]

    # Uniform weights for line seeds 
    seed_weights = np.ones(len(seed_indices))

    return {
        'seed_type': 'line',
        'seed_points': seed_points,
        'seed_normals': seed_normals,
        'seed_indices': seed_indices,
        'seed_weights': seed_weights,
        'metadata': {
            'num_lines': num_seeds,
            'target_points_per_line': line_length,
            'actual_total_points': len(seed_indices),
            'points_per_line': len(seed_indices) / max(num_seeds, 1)
        }
    }


def generate_point_seeds_fps(
    pointcloud: np.ndarray,
    num_seeds: int = 5,
    use_fps: bool = True,
    seed: int = None,
    **kwargs
) -> Dict:
    """
    Generate point seeds using FPS or random sampling.

    Args:
        pointcloud: (N, 3) array of points
        num_seeds: Number of point seeds to generate
        use_fps: If True, use FPS sampling; if False, use random sampling
        seed: Random seed for reproducibility (only used for random sampling)

    Returns:
        SeedData dict with point seed information
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(pointcloud)

    # Sample seed points
    if use_fps:
        import fpsample
        seed_indices = fpsample.bucket_fps_kdline_sampling(pointcloud, num_seeds, h=5)
        seed_indices = np.array(seed_indices, dtype=np.int64)
    else:
        seed_indices = np.random.choice(N, size=num_seeds, replace=False)

    seed_points = pointcloud[seed_indices]

    # Compute normals at seed points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    all_normals = np.asarray(pcd.normals)
    seed_normals = all_normals[seed_indices]

    # Uniform weights for point seeds
    seed_weights = np.ones(len(seed_indices))

    return {
        'seed_type': 'point',
        'seed_points': seed_points,
        'seed_normals': seed_normals,
        'seed_indices': seed_indices,
        'seed_weights': seed_weights,
        'metadata': {
            'num_seeds': num_seeds
        }
    }


def generate_point_seeds_random(
    pointcloud: np.ndarray,
    num_seeds: int = 5,
    seed: int = None,
    **kwargs
) -> Dict:
    """
    Generate point seeds using random sampling.

    Args:
        pointcloud: (N, 3) array of points
        num_seeds: Number of point seeds to generate
        seed: Random seed for reproducibility

    Returns:
        SeedData dict with point seed information
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(pointcloud)
    seed_indices = np.random.choice(N, size=min(num_seeds, N), replace=False)
    seed_points = pointcloud[seed_indices]

    # Compute normals at seed points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    all_normals = np.asarray(pcd.normals)
    seed_normals = all_normals[seed_indices]

    # Uniform weights for point seeds
    seed_weights = np.ones(len(seed_indices))

    return {
        'seed_type': 'point',
        'seed_points': seed_points,
        'seed_normals': seed_normals,
        'seed_indices': seed_indices,
        'seed_weights': seed_weights,
        'metadata': {
            'num_seeds': num_seeds
        }
    }


# Registry of seed generators
SEED_GENERATORS: Dict[str, Callable] = {
    'line_fps': generate_line_seed_fps,
    'point_fps': generate_point_seeds_fps,
    'point_random': generate_point_seeds_random,
}


def get_seed_generator(seed_type: str) -> Callable:
    """
    Get seed generator function by name.

    Args:
        seed_type: Name of seed generator (e.g., 'line_fps', 'point_fps')

    Returns:
        Seed generator function

    Raises:
        ValueError: If seed_type not found in registry
    """
    if seed_type not in SEED_GENERATORS:
        available = ', '.join(SEED_GENERATORS.keys())
        raise ValueError(f"Unknown seed type '{seed_type}'. Available: {available}")

    return SEED_GENERATORS[seed_type]
