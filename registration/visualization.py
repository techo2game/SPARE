"""Visualization utilities for point clouds and anomaly scores."""
import numpy as np
import open3d as o3d


def visualize_comparison(test_pc, reference_pc, title="Test (Red) vs Reference (Blue)"):
    """Visualize test point cloud (red) and reference (blue) overlaid."""
    # Create point clouds
    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(test_pc)
    test_pcd.paint_uniform_color([1, 0, 0])  # Red

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(reference_pc)
    ref_pcd.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize
    o3d.visualization.draw_geometries([test_pcd, ref_pcd], window_name=title)


def visualize_anomaly_scores(test_pc, anomaly_scores, template_pc=None, title="Anomaly Scores (Blue=Low, Red=High)"):
    """Visualize point cloud with anomaly scores as rainbow colormap.

    Args:
        test_pc: Point cloud (N, 3)
        anomaly_scores: Per-point anomaly scores (N,)
        template_pc: Optional template point cloud to show in green (M, 3)
        title: Window title
    """
    import matplotlib.pyplot as plt

    # Normalize scores to [0, 1]
    scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)

    # Use matplotlib's rainbow colormap (blue -> green -> yellow -> red)
    cmap = plt.get_cmap('jet')  # jet is classic rainbow: blue (low) -> red (high)
    colors = cmap(scores_norm)[:, :3]  # Get RGB, drop alpha

    # Create test point cloud with anomaly score colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(test_pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # Visualize
    o3d.visualization.draw_geometries(geometries, window_name=title)