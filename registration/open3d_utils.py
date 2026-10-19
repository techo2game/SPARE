"""Open3D registration utilities: RANSAC, ICP, FPFH, transformations."""
import numpy as np
import open3d as o3d
from copy import deepcopy as copy

try:
    from .methods.adam_chamfer_9dof import adam_chamfer_9dof
    ADAM_CHAMFER_9DOF_AVAILABLE = True
except ImportError:
    ADAM_CHAMFER_9DOF_AVAILABLE = False
    adam_chamfer_9dof = None

def compute_fpfh(pcd, voxel_size):
    """Compute FPFH features for RANSAC registration."""
    # Smaller radii for more local features (better for detailed objects like airplanes)
    radius_normal = voxel_size * 0.5  # 2.5% of diameter (was 10%)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 1.0  # 5% of diameter (was 25%)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh



def compute_voxel_size_from_scale(pcd):
    """Estimate a reasonable voxel size based on point cloud scale."""
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())
    voxel_size = diag / 100.0  # 1% of object size
    return voxel_size


def subsample_points_and_features(pcd, fpfh, fraction=0.1):
    """Subsample point cloud and corresponding FPFH features."""
    n = len(pcd.points)
    idx = np.random.choice(n, int(n * fraction), replace=False)
    pcd_sub = o3d.geometry.PointCloud()
    pcd_sub.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx])
    pcd_sub.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx]) if pcd.has_colors() else o3d.utility.Vector3dVector([])
    fpfh_sub = o3d.pipelines.registration.Feature()
    fpfh_sub.data = fpfh.data[:, idx]
    return pcd_sub, fpfh_sub


def apply_transformation(points, transform):
    """Apply 4x4 transformation matrix to points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(transform)
    return np.asarray(pcd.points)


def compute_distances(source, target, point_to_plane=False, use_fpfh=False, combine_fpfh=False, voxel_size=0.05):
    """Compute nearest neighbor distances.

    Args:
        source: Source point cloud (N, 3)
        target: Target point cloud (M, 3)
        point_to_plane: If True, compute point-to-plane distances; otherwise point-to-point
        use_fpfh: If True, compute FPFH feature distances instead of geometric distances
        combine_fpfh: If True, combine geometric + FPFH distances (weighted sum)
        voxel_size: Voxel size for FPFH computation

    Returns:
        distances: Array of distances (N,)
    """
    from sklearn.neighbors import NearestNeighbors

    # Geometric distances (point-to-point or point-to-plane)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
    distances, indices = nbrs.kneighbors(source)

    if point_to_plane:
        # Compute normals for target point cloud
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        target_normals = np.asarray(target_pcd.normals)

        # Compute point-to-plane distance for each source point
        point_to_plane_distances = []
        for i, (src_pt, nn_idx) in enumerate(zip(source, indices.squeeze())):
            nearest_pt = target[nn_idx]
            nearest_normal = target_normals[nn_idx]
            # Distance = dot product of (source - nearest) with normal
            diff = src_pt - nearest_pt
            plane_dist = np.abs(np.dot(diff, nearest_normal))
            point_to_plane_distances.append(plane_dist)

        geometric_distances = np.array(point_to_plane_distances)
    else:
        geometric_distances = distances.squeeze()

    # FPFH feature distances
    if use_fpfh or combine_fpfh:
        # Compute FPFH features for both source and target
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)
        source_fpfh = compute_fpfh(source_pcd, voxel_size)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        target_fpfh = compute_fpfh(target_pcd, voxel_size)

        # Convert FPFH to numpy arrays (shape: [feature_dim, N])
        source_features = np.asarray(source_fpfh.data).T  # [N, feature_dim]
        target_features = np.asarray(target_fpfh.data).T  # [M, feature_dim]

        # Find nearest neighbor in feature space
        nbrs_fpfh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_features)
        fpfh_distances, _ = nbrs_fpfh.kneighbors(source_features)
        fpfh_distances = fpfh_distances.squeeze()

        if use_fpfh and not combine_fpfh:
            # Use FPFH distances only
            return fpfh_distances

        # Combine geometric + FPFH distances
        if combine_fpfh:
            # Normalize both to [0, 1] before combining
            geom_norm = (geometric_distances - geometric_distances.min()) / (geometric_distances.max() - geometric_distances.min() + 1e-8)
            fpfh_norm = (fpfh_distances - fpfh_distances.min()) / (fpfh_distances.max() - fpfh_distances.min() + 1e-8)

            # Weighted sum: 50% geometric, 50% FPFH
            combined_distances = 0.5 * geom_norm + 0.5 * fpfh_norm
            return combined_distances

    return geometric_distances

    """
    Extract patches using FPS + KNN (batched, PyTorch-only implementation).

    Args:
        pointcloud: (B, N, 3) - batch of point clouds
        num_groups: number of patches per point cloud
        group_size: points per patch

    Returns:
        patches: (B, num_groups, group_size, 3) - extracted and centered patches
        knn_indices: (B, num_groups, group_size) - KNN indices for remapping scores
    """
    import torch
    try:
        import torch_fpsample
    except ImportError:
        raise ImportError("torch_fpsample not available. Install with: pip install torch-fpsample")

    B, N, _ = pointcloud.shape
    device = pointcloud.device

    # Step 1: FPS to get center points for each batch element
    # torch_fpsample.sample returns (sampled_points, indices)
    fps_centers, fps_idx = torch_fpsample.sample(pointcloud, num_groups, start_idx=0)  # fps_centers: (B, G, 3), fps_idx: (B, G)

    # Step 2: KNN using torch.cdist to find nearest neighbors for each FPS center
    # Compute pairwise distances: (B, G, N)
    dists = torch.cdist(fps_centers, pointcloud, p=2)  # (B, num_groups, N)

    # Get top-K nearest neighbors (smallest distances)
    _, knn_indices = torch.topk(dists, group_size, dim=2, largest=False)  # (B, num_groups, group_size)

    # Step 3: Extract patches using gathered indices
    # Expand indices to match pointcloud dimensions for gathering
    # knn_indices: (B, G, K) -> (B, G, K, 3)
    knn_indices_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)

    # Gather points: (B, N, 3) -> (B, G, K, 3)
    patches = torch.gather(pointcloud.unsqueeze(1).expand(-1, num_groups, -1, -1),
                          2,
                          knn_indices_expanded)  # (B, num_groups, group_size, 3)

    # Step 4: Normalize each patch by centering at its FPS center
    fps_centers_expanded = fps_centers.unsqueeze(2)  # (B, G, 1, 3)
    patches = patches - fps_centers_expanded  # (B, G, K, 3)

    # Ensure float32 dtype (important for compatibility with GPU operations)
    patches = patches.float()

    return patches, knn_indices



    """Extract patches using FPS + KNN.

    Args:
        pointcloud: numpy array (N, 3)
        num_groups: number of patches (FPS centers)
        group_size: number of points per patch (KNN neighbors)

    Returns:
        patches: list of numpy arrays, each (group_size, 3)
        knn_indices: KNN indices (num_groups, group_size) for remapping scores
    """
    import fpsample
    from sklearn.neighbors import KDTree

    # FPS to get center points using fpsample (100× faster than pointnet2)
    fps_idx = fpsample.bucket_fps_kdline_sampling(pointcloud, num_groups, h=5)
    center_points = pointcloud[fps_idx]

    # KNN using sklearn KDTree (much faster than KNN_CUDA for this use case)
    tree = KDTree(pointcloud, leaf_size=30)
    _, knn_indices = tree.query(center_points, k=group_size)

    # Extract patches and normalize (center at origin)
    patches = []
    for g in range(num_groups):
        patch_indices = knn_indices[g]
        patch_points = pointcloud[patch_indices]
        # Normalize: center at origin
        patch_centered = patch_points - center_points[g]
        patches.append(patch_centered)

    # Return patches and KNN indices for score remapping
    return patches, knn_indices