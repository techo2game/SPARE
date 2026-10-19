"""
Mask generation functions for synthetic anomaly detection.

Point cloud-based mask generation using KNN line components.
"""
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import fpsample



def generate_fps_seeds(pointcloud, num_seeds, seed=None):
    """
    Generate seed points using Farthest Point Sampling.

    Args:
        pointcloud: (N,3) array
        num_seeds: number of seed points to sample
        seed: random seed

    Returns:
        seed_indices: (num_seeds,) array of indices
    """
    fps_indices = fpsample.bucket_fps_kdline_sampling(pointcloud, num_seeds, h=3)
    return fps_indices


def generate_knn_line_component(pointcloud, seed_idx, k_line=5, line_length=10, k_min=10,
                                k_amplitude=50, num_best_directions=10, seed=None):
    """
    Generate a 3D component starting as a line by iteratively hopping to nearest neighbors,
    then expand the line using variable-width KNN growth based on cosine weighting.

    One full cosine period is mapped across the line length for smooth thickness variation.

    Args:
        pointcloud: (N,3) array
        seed_idx: index of starting point
        k_line: number of nearest neighbors to consider when growing the initial line
        line_length: number of hops along the line
        k_min: minimum number of KNN neighbors
        k_amplitude: amplitude of cosine oscillation
        num_best_directions: number of top-aligned directions to randomly choose from
        seed: random seed

    Returns:
        component_indices: array of all point indices in the component
        line_spine_indices: array of line spine point indices
        line_spine_k_values: array of k values for each spine point (for gaussian_z weighting)
    """
    rng = np.random.default_rng(seed)
    N = len(pointcloud)
    visited = set([seed_idx])

    nbrs_line = NearestNeighbors(n_neighbors=k_line + 1, algorithm='auto').fit(pointcloud)

    # --- Generate line by hopping to nearest neighbors with direction continuity ---
    current_idx = seed_idx
    line_points = [current_idx]
    prev_direction = None

    for _ in range(line_length - 1):
        distances, neighbors = nbrs_line.kneighbors(pointcloud[current_idx].reshape(1, -1))
        neighbors = neighbors[0]
        # choose a neighbor not yet visited
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if not unvisited_neighbors:
            break

        # If we have a previous direction, choose randomly from neighbors that continue in that direction
        if prev_direction is not None and len(line_points) >= 2:
            # Compute direction to each candidate neighbor
            current_pos = pointcloud[current_idx]
            candidate_directions = pointcloud[unvisited_neighbors] - current_pos
            # Normalize
            norms = np.linalg.norm(candidate_directions, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            candidate_directions = candidate_directions / norms

            # Compute dot product with previous direction (higher = more aligned)
            alignment = candidate_directions @ prev_direction

            # Select top candidates based on alignment with previous direction
            top_k = max(1, min(num_best_directions, len(alignment)))
            top_indices = np.argsort(alignment)[-top_k:]  # Indices of top K aligned

            # Choose randomly from the top candidates
            chosen_idx = rng.choice(top_indices)
            next_idx = unvisited_neighbors[chosen_idx]

            # Update direction
            prev_direction = (pointcloud[next_idx] - current_pos) / np.linalg.norm(pointcloud[next_idx] - current_pos)
        else:
            # First hop: choose randomly (or pick the nearest unvisited)
            next_idx = unvisited_neighbors[0]  # pick closest
            if len(line_points) >= 1:
                # Set initial direction
                prev_direction = pointcloud[next_idx] - pointcloud[current_idx]
                prev_direction = prev_direction / np.linalg.norm(prev_direction)

        visited.add(next_idx)
        line_points.append(next_idx)
        current_idx = next_idx

    # --- Variable-width KNN expansion using cosine weighting ---
    line_positions = np.arange(len(line_points))

    # Map line positions to [0, π] for one full cosine period
    # Cosine function mapped to [0, 1]: weight = 0.5 * (1 + cos(theta))
    theta = 1 * np.pi * line_positions / len(line_points)
    cosine_weights = 0.5 * (1 + np.cos(theta))

    # Safety: clamp amplitude to avoid requesting more neighbors than exist
    max_possible_k = len(pointcloud) - 1
    k_amplitude_safe = min(k_amplitude, max_possible_k - k_min)

    # k_values = k_min + amplitude * weight  (weight in [0, 1])
    k_values = (k_min + k_amplitude_safe * cosine_weights).astype(int)
    # Ensure minimum of 1 neighbor
    k_values = np.maximum(k_values, 1)

    # Expand each line point with its specific k value
    # Build KNN index once with maximum k
    max_k = k_values.max()
    nbrs_expand = NearestNeighbors(n_neighbors=max_k, algorithm='auto').fit(pointcloud)

    for i, line_pt_idx in enumerate(line_points):
        k_this_point = k_values[i]
        # Get k nearest neighbors for this specific line point
        _, neighbors = nbrs_expand.kneighbors(pointcloud[line_pt_idx].reshape(1, -1), n_neighbors=k_this_point)
        neighbors = neighbors[0]

        for n in neighbors:
            if n not in visited:
                visited.add(n)

    # Return component indices, spine indices, and spine k values for gaussian_z weighting
    component_indices = np.array(list(visited))
    line_spine_indices = np.array(line_points)
    line_spine_k_values = k_values

    return component_indices, line_spine_indices, line_spine_k_values


def generate_multiple_knn_line_components(pointcloud, num_components=5, k_line=5, line_length=10,
                                          k_min=10, k_amplitude=50, num_best_directions=10,
                                          use_fps=True, seed=None, weighted_mask=False):
    """
    Generate multiple variable-width line components in a point cloud using connected components.

    Args:
        pointcloud: (N,3) array
        num_components: number of separate line components to generate
        k_line: number of neighbors to consider when growing each line
        line_length: number of hops along each line
        k_min: minimum number of neighbors
        k_amplitude: amplitude of cosine oscillation
        num_best_directions: number of top-aligned directions to randomly choose from
        use_fps: if True, use FPS for seed selection; otherwise random
        seed: random seed
        weighted_mask: if True, mask values are weighted by spine K values [0,1]; if False, binary [0,1]

    Returns:
        mask: (N,) float array (weighted or binary) indicating anomalous points
        component_ids: (N,) int array with component IDs (0 = background)
        component_data: list of dicts with keys:
            'indices': point indices in component
            'spine_indices': line spine point indices
            'spine_k_values': k values for each spine point (for gaussian_z weighting)
    """
    N = len(pointcloud)
    rng = np.random.default_rng(seed)
    mask = np.zeros(N, dtype=np.float32)  # Always float to support both binary and weighted
    component_ids = np.zeros(N, dtype=np.int32)
    component_data = []
    available_points = set(range(N))

    # Generate seed points using FPS or random
    if use_fps:
        fps_seeds = generate_fps_seeds(pointcloud, num_components, seed=seed)
        seed_indices = [int(idx) for idx in fps_seeds if idx in available_points]
    else:
        seed_indices = []

    for comp_id in range(1, num_components+1):
        if not available_points:
            break

        # Select seed
        if use_fps and seed_indices:
            seed_idx = seed_indices[comp_id - 1] if (comp_id - 1) < len(seed_indices) else rng.choice(list(available_points))
        else:
            seed_idx = rng.choice(list(available_points))

        # Generate component
        comp_indices, spine_indices, spine_k_values = generate_knn_line_component(
            pointcloud, seed_idx,
            k_line=k_line,
            line_length=line_length,
            k_min=k_min,
            k_amplitude=k_amplitude,
            num_best_directions=num_best_directions,
            seed=seed
        )

        component_ids[comp_indices] = comp_id
        available_points -= set(comp_indices)

        # Store component metadata for deformation methods
        component_data.append({
            'component_id': comp_id,
            'indices': comp_indices,
            'spine_indices': spine_indices,
            'spine_k_values': spine_k_values
        })

    return mask, component_ids, component_data


def generate_anomaly_mask_knn_lines(pointcloud, num_components=15, k_line=200, line_length=20,
                                    k_min=15, k_amplitude=600, num_best_directions=10,
                                    use_fps=True, seed=42, weighted_mask=False, **kwargs):
    """
    Generate anomaly mask using KNN line components with connected components.

    Creates smooth, variable-width line anomalies by growing lines through the point cloud
    using directional KNN hops, then expanding with cosine-weighted thickness variation.

    Args:
        pointcloud: (N, 3) numpy array
        num_components: number of separate line components to generate
        k_line: number of neighbors to consider when growing each line
        line_length: number of hops along the line spine (int or float 0-1 for percentage of N)
        k_min: minimum number of neighbors (minimum line thickness) (int or float 0-1 for percentage of N)
        k_amplitude: amplitude of cosine oscillation (thickness variation) (int or float 0-1 for percentage of N)
        num_best_directions: number of top-aligned directions to randomly choose from
        use_fps: if True, use FPS for seed selection; otherwise random
        seed: random seed for reproducibility
        weighted_mask: if True, mask values weighted by spine K values [0,1]; if False, binary
        **kwargs: additional arguments (for API compatibility)

    Returns:
        mask: (N,) float array indicating anomalous points (weighted or binary)
        metadata: dict with additional information:
            'component_ids': (N,) int array with component IDs (0 = background)
            'component_data': list of dicts with:
                'indices': point indices in component
                'spine_indices': line spine point indices
                'spine_k_values': k values for each spine point (for gaussian_z weighting)
    """
    # Convert percentage-based parameters to absolute values
    N = len(pointcloud)

    line_length = max(1, int(line_length * N))
    k_min = max(1, int(k_min * N))
    k_amplitude = max(1, int(k_amplitude * N))

    mask, component_ids, component_data = generate_multiple_knn_line_components(
        pointcloud,
        num_components=num_components,
        k_line=k_line,
        line_length=line_length,
        k_min=k_min,
        k_amplitude=k_amplitude,
        num_best_directions=num_best_directions,
        use_fps=use_fps,
        seed=seed,
        weighted_mask=weighted_mask
    )

    # Return mask and metadata
    metadata = {
        'component_ids': component_ids,
        'component_data': component_data
    }

    return mask, metadata


def generate_focused_knn_line(pointcloud, seed_idx, num_points=100, curvature=0.3):
    """
    Generate a simple KNN line with controlled curvature.
    
    This is the focused approach with just points and curvature control.
    
    Args:
        pointcloud: (N, 3) array
        seed_idx: index of starting point
        num_points: number of points in the line (absolute, not percentage)
        curvature: 0.0 (straight) to 1.0 (very curved)
    
    Returns:
        component_indices: array of point indices in the line
        spine_indices: array of spine point indices
        spine_k_values: array of k values for spine points
    """
    # Start with seed point
    current_idx = seed_idx
    line_points = [current_idx]
    visited = set([current_idx])
    
    # Build KNN index
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(pointcloud)
    
    # Generate line with curvature
    for _ in range(num_points - 1):
        # Get neighbors of current point
        _, neighbors = nbrs.kneighbors(pointcloud[current_idx].reshape(1, -1), n_neighbors=51)
        neighbors = neighbors[0][1:]  # Exclude self
        
        # Filter unvisited neighbors
        unvisited = [n for n in neighbors if n not in visited]
        if not unvisited:
            break
        
        # Add some curvature by occasionally choosing non-straight directions
        if np.random.rand() < curvature:
            # Choose a more curved direction
            current_idx = np.random.choice(unvisited)
        else:
            # Choose straightest direction (closest to current direction)
            current_pos = pointcloud[current_idx]
            neighbor_pos = pointcloud[unvisited]
            vecs = neighbor_pos - current_pos
            if len(vecs) > 0:
                # Find neighbor most aligned with current direction
                if len(line_points) >= 2:
                    prev_dir = pointcloud[line_points[-1]] - pointcloud[line_points[-2]]
                    prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-8)
                    alignments = np.dot(vecs, prev_dir)
                    current_idx = unvisited[np.argmax(alignments)]
                else:
                    current_idx = unvisited[0]
            else:
                current_idx = unvisited[0]
        
        visited.add(current_idx)
        line_points.append(current_idx)
    
    # Create spine with simple K values
    spine_indices = line_points[::len(line_points)//5]  # Every 5th point
    spine_k_values = np.ones(len(spine_indices)) * 10  # Simple fixed K values
    
    return np.array(line_points, dtype=np.int32), np.array(spine_indices, dtype=np.int32), spine_k_values


def generate_focused_components(pointcloud, num_components=3, num_points=100, curvature=0.3):
    """
    Generate multiple focused KNN line components using FPS for placement.
    
    This is the simple approach requested:
    - num_components: how many lines
    - num_points: points per line (absolute number)
    - curvature: how curved the lines are
    - FPS for good seed placement
    
    Args:
        pointcloud: (N, 3) array
        num_components: number of line components
        num_points: number of points per line (absolute)
        curvature: curvature amount (0.0-1.0)
    
    Returns:
        mask: (N,) binary mask
        component_ids: (N,) component IDs
        component_data: list of component metadata
    """
    N = len(pointcloud)
    mask = np.zeros(N, dtype=np.float32)
    component_ids = np.zeros(N, dtype=np.int32)
    component_data = []
    
    # Use FPS to find good seed points
    try:
        import fpsample
        fps_indices = fpsample.bucket_fps_kdline_sampling(pointcloud, num_components, h=3)
    except ImportError:
        # Fallback to random if fpsample not available
        fps_indices = np.random.choice(len(pointcloud), num_components, replace=False)
    
    for comp_id in range(1, num_components + 1):
        if len(fps_indices) < comp_id:
            break
            
        seed_idx = fps_indices[comp_id - 1]
        
        # Generate focused KNN line
        comp_indices, spine_indices, spine_k_values = generate_focused_knn_line(
            pointcloud, seed_idx, num_points, curvature
        )
        
        # Mark component (ensure comp_indices is integer array)
        comp_indices_int = comp_indices.astype(int)
        component_ids[comp_indices_int] = comp_id
        mask[comp_indices_int] = 1.0  # Fill the mask!
        
        # Store metadata
        component_data.append({
            'component_id': comp_id,
            'indices': comp_indices,
            'spine_indices': spine_indices,
            'spine_k_values': spine_k_values
        })
    
    return mask, component_ids, component_data


def generate_anomaly_mask_focused_lines(pointcloud, num_components=3, num_points=100, 
                                       curvature=0.3, **kwargs):
    """
    Generate anomaly mask using focused KNN lines.
    
    Simple approach with just:
    - num_components: number of lines
    - num_points: points per line (absolute)
    - curvature: how curved (0.0-1.0)
    
    Args:
        pointcloud: (N, 3) numpy array
        num_components: number of line components
        num_points: number of points per line (absolute)
        curvature: curvature amount (0.0-1.0)
        **kwargs: additional arguments (for API compatibility)
    
    Returns:
        mask: (N,) float array indicating anomalous points
        metadata: dict with additional information
    """
    mask, component_ids, component_data = generate_focused_components(
        pointcloud, num_components, num_points, curvature
    )
    
    metadata = {
        'component_ids': component_ids,
        'component_data': component_data
    }
    
    return mask, metadata


# Registry of available mask generators
MASK_GENERATORS = {
    'knn_lines': generate_anomaly_mask_knn_lines,
    'focused_lines': generate_anomaly_mask_focused_lines,
}


def get_mask_generator(mask_type):
    """
    Get mask generator function by name.

    Args:
        mask_type: str, 'knn_lines' or 'focused_lines'

    Returns:
        function: Mask generator function returning (mask, metadata) tuple
    """
    if mask_type not in MASK_GENERATORS:
        raise ValueError(f"Unknown mask type: {mask_type}. Available: {list(MASK_GENERATORS.keys())}")
    return MASK_GENERATORS[mask_type]