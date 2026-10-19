"""
Script 2: Generate synthetic anomalies from training samples.

Simple script that:
1. Loads training point clouds
2. Generates synthetic anomalies (bulges, sinks, distortions)
3. Saves: anomalous point cloud + ground truth mask

All parameters (num_synthetic_per_sample, severity, amplitude, etc.)
are configured in configs/
"""
import os
import sys
import argparse
import numpy as np
import pickle
import open3d as o3d
import torch
from tqdm import tqdm
from easydict import EasyDict

from utils.dataset import *
from utils.misc import seed_all
from utils.config import cfg_from_yaml_file
from utils.synthetic_anomalies import *
from utils.artifacts import dataset_dir_name
from utils.data_loading import list_train_files




def visualize_anomaly(points, mask, title="Synthetic Anomaly"):
    """
    Visualize point cloud with anomaly mask.
    Red = normal (0), Blue = anomaly (1)
    """
    print(f"\n[VIZ] Opening visualization: {title}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((len(points), 3))
    colors[mask == 0] = [1, 0, 0]  # Red for normal
    colors[mask == 1] = [0, 0, 1]  # Blue for anomaly

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
        width=1024,
        height=768,
        point_show_normal=False
    )
    print(f"[VIZ] Window closed: {title}")


def normalize_shape_bbox(pc):
    """Normalize point cloud using shape_bbox mode."""
    pc_max = pc.max(axis=0, keepdims=True)
    pc_min = pc.min(axis=0, keepdims=True)
    shift = (pc_min + pc_max) / 2
    scale = (pc_max - pc_min).max() / 2
    pc_normalized = (pc - shift) / scale
    return pc_normalized


def load_point_cloud(file_path, is_mvtec=False):
    """Load point cloud from file."""
    if is_mvtec:
        import tifffile as tiff
        depth_image = tiff.imread(file_path)
        valid_mask = depth_image[:, :, 2] != 0
        points = depth_image[valid_mask].astype(np.float32)
    else:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.array(pcd.points, dtype=np.float32)

    return points


def generate_synthetic_anomalies(args):

    # Load dataset config
    dataset_config = EasyDict()
    cfg_from_yaml_file(args.config, dataset_config)

    # Load synthetic anomaly config
    synth_config = EasyDict()
    cfg_from_yaml_file(args.synth_config, synth_config)

    # Override visualize from CLI if specified
    if args.visualize:
        synth_config.visualize = True

    dataset_name = dataset_config.dataset
    dataset_path = dataset_config.dataset_path
    category = args.category


    seed_all(synth_config.seed)

    dataset_dir = dataset_dir_name(dataset_name)
    output_dir = os.path.join(synth_config.output_dir, dataset_dir, category)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCategory: {category}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_dir}")

    # Training files (sample i was registered to reg_output/.../sample_{i:03d}.pkl)
    _, train_files = list_train_files(dataset_path, category, dataset_name)

    print(f"Found {len(train_files)} training samples")

    # Generate synthetic anomalies
    total_synthetic = len(train_files) * synth_config.num_synthetic_per_sample


    synthetic_count = 0

    for sample_idx, train_file in enumerate(tqdm(train_files, desc="Processing samples")):
        # Load aligned point cloud from reg_output
        aligned_file = f'sample_{sample_idx:03d}.pkl'
        aligned_path = os.path.join('reg_output', dataset_dir, category, 'train', aligned_file)
        if not os.path.exists(aligned_path):
            raise FileNotFoundError(
                f"{aligned_path} not found. Run compute_train_variability.py "
                "--use_parenet --save_transforms first."
            )

        # Load pre-aligned data
        with open(aligned_path, 'rb') as f:
            aligned_data = pickle.load(f)
        points_norm = aligned_data['points']  # Already aligned to sample 0

        # Helper function to cut point cloud for Real3DAD
        def cut_real3dad_half(points):
            """Cut point cloud in half along y-dimension with random position (Real3DAD only)"""
            return points
            if dataset_name != 'Real3DAD':
                return points
            dim = 1
            # Randomly vary the cut position between 30% and 70% of the range
            threshold = np.random.uniform(0.3, 0.7)
            dim_min = points[:, dim].min()
            dim_max = points[:, dim].max()
            cut_position = dim_min + threshold * (dim_max - dim_min)
            # Randomly choose upper or lower side
            if np.random.rand() < 0.5:
                # Keep upper half
                mask_half = points[:, dim] >= cut_position
            else:
                # Keep lower half
                mask_half = points[:, dim] < cut_position
            return points[mask_half]

        # Helper function to get config for deformation type
        def get_config_for_deform(deform_config_entry):
            """
            Get config for deformation. Supports two formats:
            1. Old format: {name: 'gaussian_z', count: 10} - uses hardcoded mapping
            2. New format: {name: 'gaussian_z', config: 'aug1', count: 10} - uses explicit config section
            """
            deform_type = deform_config_entry['name']

            # If 'config' key is explicitly provided, use it directly
            if 'config' in deform_config_entry:
                config_key = deform_config_entry['config']
            else:
                # Fall back to hardcoded mapping for backward compatibility
                config_map = {
                    'component_normal': 'component_normal_config',
                    'gaussian_force': 'gaussian_force_config',
                    'rbf': 'rbf_config',
                    'rbf_inv': 'rbf_config',
                }
                config_key = config_map.get(deform_type, deform_type)

            if config_key not in synth_config:
                raise ValueError(f"Config not found for {deform_type} (looked for '{config_key}')")
            return synth_config[config_key], deform_type

        # Generate TRAINING samples
        for deform_config in synth_config.train_deformations:
            deform_type = deform_config['name']
            count = deform_config['count']
            generator_config, deform_type = get_config_for_deform(deform_config)

            for _ in range(count):
                # For Real3DAD: cut with random position for each variation
                points_cut = cut_real3dad_half(points_norm.copy())

                anomaly_gen = AnomalyGenerator(generator_config)
                anomalous_points, mask = anomaly_gen.compute(points_cut)

                synthetic_data = {
                    'points': torch.as_tensor(anomalous_points),
                    'mask': torch.as_tensor(mask),
                    'anomaly_type': deform_type,
                    'split': 'train',
                }

                save_path = os.path.join(
                    output_dir,
                    f'sample_{sample_idx:03d}_synth_{synthetic_count:03d}_{deform_type}.pkl'
                )
                torch.save(synthetic_data, save_path)
                synthetic_count += 1

                if synth_config.get('visualize', False):
                    visualize_anomaly(anomalous_points, mask, f"{train_file} - TRAIN - {deform_type}")

        # Generate VALIDATION samples
        for deform_config in synth_config.val_deformations:
            deform_type = deform_config['name']
            count = deform_config['count']
            generator_config, deform_type = get_config_for_deform(deform_config)

            for _ in range(count):
                # For Real3DAD: cut with random position for each variation
                points_cut = cut_real3dad_half(points_norm.copy())

                anomaly_gen = AnomalyGenerator(generator_config)
                anomalous_points, mask = anomaly_gen.compute(points_cut)

                synthetic_data = {
                    'points': torch.as_tensor(anomalous_points),
                    'mask': torch.as_tensor(mask),
                    'anomaly_type': deform_type,
                    'split': 'val',
                }

                save_path = os.path.join(
                    output_dir,
                    f'sample_{sample_idx:03d}_synth_{synthetic_count:03d}_{deform_type}.pkl'
                )
                torch.save(synthetic_data, save_path)
                synthetic_count += 1

                if synth_config.get('visualize', False):
                    visualize_anomaly(anomalous_points, mask, f"{train_file} - VAL - {deform_type}")

    print(f"\nGeneration complete: {synthetic_count} total synthetic samples")
    print(f"Saved to: {output_dir}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Dataset config file (e.g., configs/mvtec3d-ad/base.yaml)')
    parser.add_argument('--category', type=str, required=True, help='Category name')
    parser.add_argument('--synth_config', type=str, default='configs/synthetic_anomalies_seed_based.yaml', help='Synthetic anomaly config file')
    parser.add_argument('--visualize', action='store_true', help='Visualize generated anomalies')
    args = parser.parse_args()

    generate_synthetic_anomalies(args)


if __name__ == '__main__':
    main()