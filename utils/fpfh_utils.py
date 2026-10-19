"""
FPFH (Fast Point Feature Histograms) features, computed with Open3D.
"""
import numpy as np
import open3d as o3d
import torch

# Search radii as multiples of the voxel size (SPARE uses voxel_size = 5% of the object diameter)
RADIUS_NORMAL = 2.0
RADIUS_FEATURE = 5.0


def compute_fpfh_features(points, voxel_size, max_nn_normal=100, max_nn_feature=100):
    """
    Compute FPFH features.

    Args:
        points: Point cloud (N, 3), torch tensor (any device) or numpy array
        voxel_size: Voxel size; normals use 2x and features 5x this radius
        max_nn_normal: Max neighbors for normal estimation
        max_nn_feature: Max neighbors for the FPFH histograms

    Returns:
        FPFH features (N, 33), same type (and device) as points
    """
    is_torch = isinstance(points, torch.Tensor)
    points_np = points.detach().cpu().numpy() if is_torch else np.asarray(points)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np.astype(np.float64)))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_NORMAL * voxel_size, max_nn=max_nn_normal)
    )
    # Consistent sign: normals point toward the origin (point clouds are centered)
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=RADIUS_FEATURE * voxel_size, max_nn=max_nn_feature),
    )
    features = np.asarray(fpfh.data, dtype=np.float32).T

    if is_torch:
        return torch.from_numpy(features).to(points.device)
    return features
