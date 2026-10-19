"""Compute variability between training samples within each category.

This script measures intra-class variance by:
1. Loading all training samples for a category
2. Computing pairwise registration quality (fitness, RMSE) between all training samples
3. Computing pairwise point-to-point distances after registration
4. Reporting statistics (mean, std, min, max) for each category

Usage:
    # Single category with PARENet
    python compute_train_variability.py configs/real3d-ad/base.yaml --category airplane --use_parenet

    # All categories with PARENet
    python compute_train_variability.py configs/real3d-ad/base.yaml --use_parenet

    # With Adam 9DOF
    python compute_train_variability.py configs/real3d-ad/base.yaml --use_adam_9dof

    # Save transformations for later use
    python compute_train_variability.py configs/real3d-ad/base.yaml --use_parenet --save_transforms --first_sample_only
"""
import sys
import argparse
import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict
from itertools import combinations
from tqdm import tqdm
import os

from utils.dataset import *
from utils.misc import seed_all
from utils.config import cfg_from_yaml_file
from registration.open3d_utils import (
    compute_fpfh,
    adam_chamfer_9dof,
    apply_transformation,
    compute_distances
)
from registration.methods.parenet_torch import parenet_torch
from utils.artifacts import dataset_dir_name
from utils.data_loading import list_train_files, load_train_pointcloud



def normalize_shape_bbox(pc):
    """Normalize point cloud using shape_bbox mode.

    Centers at bbox center and scales so the longest dimension fits in [-1, 1].
    Other dimensions scale proportionally.

    Args:
        pc: numpy array (N, 3)

    Returns:
        pc_normalized: numpy array (N, 3)
    """
    pc_max = pc.max(axis=0, keepdims=True)  # (1, 3)
    pc_min = pc.min(axis=0, keepdims=True)  # (1, 3)
    shift = (pc_min + pc_max) / 2  # Center at bbox center
    scale = (pc_max - pc_min).max() / 2  # Half the max dimension
    pc_normalized = (pc - shift) / scale
    return pc_normalized

def visualize_registration(source_points, target_points, title="Registration Result"):
    """Visualize two point clouds after registration.

    Args:
        source_points: Source point cloud as numpy array (N, 3)
        target_points: Target point cloud as numpy array (M, 3)
        title: Window title
    """
    # Create Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    source_pcd.paint_uniform_color([1, 0, 0])  # Red

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize
    o3d.visualization.draw_geometries(
        [source_pcd, target_pcd],
        window_name=title,
        width=1024,
        height=768
    )

def compute_pairwise_variability(train_files, train_path, args, category=None, dataset=None):
    """Compute pairwise registration quality and distances between training samples.

    Args:
        train_files: List of training .pcd filenames
        train_path: Path to training directory
        args: Arguments with registration parameters
        category: Category name for saving results
        dataset: Dataset name from config (e.g., 'ShapeNetAD', 'Real3DAD')

    Returns:
        dict with statistics
    """
    # Create output directory for this category
    if args.save_transforms and category:
        output_dir = os.path.join('reg_output', dataset_dir_name(dataset), category)
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving transformations to: {output_dir}")

    # Load all training samples
    train_templates = []
    train_pcds = []
    train_fpfhs = []
    train_diameters = []

    print(f"  Loading {len(train_files)} training samples...")

    for train_file in train_files:
        train_file_path = os.path.join(train_path, train_file)

        # PCD (Real3DAD/ShapeNetAD) or organized TIFF (MVTec3D-AD, valid points only)
        train_template = load_train_pointcloud(train_file_path)

        # Normalize using shape_bbox (same as dataset)
        train_template = normalize_shape_bbox(train_template)

        # Downsample to max_points
        if len(train_template) > args.max_points:
            indices = np.random.choice(len(train_template), args.max_points, replace=False)
            train_template = train_template[indices]

        # Compute diameter and voxel size
        train_pcd = o3d.geometry.PointCloud()
        train_pcd.points = o3d.utility.Vector3dVector(train_template)

        diameter = np.linalg.norm(
            np.asarray(train_pcd.points).max(axis=0) -
            np.asarray(train_pcd.points).min(axis=0)
        )

        train_templates.append(train_template)
        train_pcds.append(train_pcd)
        train_diameters.append(diameter)


    # Save reference sample (sample 0) if requested
    if args.save_transforms and category:
        train_output_dir = os.path.join(output_dir, 'train')
        os.makedirs(train_output_dir, exist_ok=True)

        import pickle
        pkl_path = os.path.join(train_output_dir, 'sample_000.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'points': train_templates[0],  # Reference (no transformation)
                'transform': np.eye(4)  # Identity transform
            }, f)
        print(f"  Saved reference sample to: {pkl_path}")

    # Compute all pairwise combinations or only pairs with first sample
    n_samples = len(train_templates)
    if args.first_sample_only:
        # Only compute pairs (i, 0) for i = 1, 2, ..., n-1
        # Register each sample TO sample 0 (not FROM sample 0)
        pairs = [(i, 0) for i in range(1, n_samples)]
        print(f"  Computing {len(pairs)} pairwise registrations (align to first sample)...")
    else:
        # Compute all pairs
        pairs = list(combinations(range(n_samples), 2))
        print(f"  Computing {len(pairs)} pairwise registrations (all pairs)...")

    fitness_scores = []
    rmse_scores = []
    mean_distances = []

    for i, j in tqdm(pairs, desc="  Pairs"):
        try:
            # Compute adaptive voxel size
            avg_diameter = (train_diameters[i] + train_diameters[j]) / 2
            adaptive_voxel_size = avg_diameter * args.ransac_voxel_size

            # Create copies for registration
            test_pcd_copy = o3d.geometry.PointCloud()
            test_pcd_copy.points = o3d.utility.Vector3dVector(train_templates[i].copy())

            train_pcd_copy = o3d.geometry.PointCloud()
            train_pcd_copy.points = o3d.utility.Vector3dVector(train_templates[j].copy())

            if args.use_parenet:
                # PARENet deep learning registration
                test_tensor = torch.from_numpy(train_templates[i]).float().cuda()
                train_tensor = torch.from_numpy(train_templates[j]).float().cuda()
                transform_tensor = parenet_torch(
                    test_tensor, train_tensor,
                    checkpoint_path=args.parenet_checkpoint
                )
                global_transform = transform_tensor.cpu().numpy()
            elif args.use_adam_9dof:
                # Adam + Chamfer 9DOF - rotation + translation optimization
                global_transform = adam_chamfer_9dof(
                    test_pcd_copy, train_pcd_copy,
                    max_iter=200,
                    num_points=4000,
                    lr=0.01,
                    use_scheduler=True,
                    num_restarts=50
                )

            # Save transformation to first object (j=0)
            if args.save_transforms and category and j == 0:
                # Create train subdirectory
                train_output_dir = os.path.join(output_dir, 'train')
                os.makedirs(train_output_dir, exist_ok=True)

                # Apply transformation to get aligned points
                test_aligned = apply_transformation(train_templates[i], global_transform)

                # Save as pickle file (for synthetic anomaly generation)
                pkl_filename = f"sample_{i:03d}.pkl"
                pkl_path = os.path.join(train_output_dir, pkl_filename)
                import pickle
                with open(pkl_path, 'wb') as f:
                    pickle.dump({
                        'points': test_aligned,  # Aligned point cloud
                        'transform': global_transform  # Transformation matrix
                    }, f)

            # Apply transformation and evaluate
            test_aligned = apply_transformation(train_templates[i], global_transform)

            # Compute fitness using Open3D evaluation
            test_aligned_pcd = o3d.geometry.PointCloud()
            test_aligned_pcd.points = o3d.utility.Vector3dVector(test_aligned)
            train_ref_pcd = o3d.geometry.PointCloud()
            train_ref_pcd.points = o3d.utility.Vector3dVector(train_templates[j])

            evaluation = o3d.pipelines.registration.evaluate_registration(
                test_aligned_pcd, train_ref_pcd,
                adaptive_voxel_size * 1.5,
                np.eye(4)
            )

            fitness = evaluation.fitness
            rmse = evaluation.inlier_rmse

            # Compute point-to-point distances
            distances = compute_distances(test_aligned, train_templates[j])
            mean_dist = np.mean(distances)

            fitness_scores.append(fitness)
            rmse_scores.append(rmse)
            mean_distances.append(mean_dist)

            # Visualize if requested
            if args.visualize:
                visualize_registration(
                    test_aligned,
                    train_templates[j],
                    title=f"Pair ({i}, {j}): Fitness={fitness:.4f}, RMSE={rmse:.6f}, Dist={mean_dist:.6f}"
                )

        except Exception as e:
            print(f"    WARNING: Pair ({i}, {j}) failed: {e}")
            continue

    if not fitness_scores:
        print("  ERROR: All pairwise registrations failed!")
        return None

    # Compute statistics
    results = {
        'n_samples': n_samples,
        'n_pairs': len(pairs),
        'n_successful': len(fitness_scores),
        'fitness_mean': np.mean(fitness_scores),
        'fitness_std': np.std(fitness_scores),
        'fitness_min': np.min(fitness_scores),
        'fitness_max': np.max(fitness_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'rmse_min': np.min(rmse_scores),
        'rmse_max': np.max(rmse_scores),
        'dist_mean': np.mean(mean_distances),
        'dist_std': np.std(mean_distances),
        'dist_min': np.min(mean_distances),
        'dist_max': np.max(mean_distances),
    }

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--category', type=str, default=None, help='Single category (if None, runs all)')
    # Registration method
    parser.add_argument('--use_adam_9dof', action='store_true', help='Use Adam + Chamfer 9DOF (rotation + translation optimization)')
    parser.add_argument('--use_parenet', action='store_true', help='Use PARENet deep learning registration')
    parser.add_argument('--parenet_checkpoint', type=str, default='pretrain/parenet.tar', help='PARENet checkpoint path')
    # RANSAC parameters
    parser.add_argument('--ransac_voxel_size', type=float, default=0.05, help='Voxel size multiplier')
    parser.add_argument('--ransac_max_iter', type=int, default=100000, help='Maximum RANSAC iterations')
    # Dataset parameters
    parser.add_argument('--max_points', type=int, default=150000, help='Max points per sample')

    # Other
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--output', type=str, default='train_variability.csv', help='Output CSV file')
    parser.add_argument('--visualize', action='store_true', help='Visualize each pairwise registration')
    parser.add_argument('--save_transforms', action='store_true', help='Save transformation matrices and registered point clouds to reg_output/')
    parser.add_argument('--first_sample_only', action='store_true', help='Only compute pairs with first sample (reduces O(n²) to O(n))')
    args = parser.parse_args()
    if args.save_transforms and not args.first_sample_only:
        # Saved samples are the ones registered TO sample 0, i.e. the (i, 0) pairs.
        # The all-pairs mode never produces them, which would leave reg_output/
        # with only sample_000 and silently build one-sample memory banks.
        print("--save_transforms: enabling --first_sample_only (registration to sample 0)")
        args.first_sample_only = True

    # Load config
    config = EasyDict()
    cfg_from_yaml_file(args.config, config)

    dataset = config.dataset
    dataset_path = config.dataset_path

    seed_all(args.seed)

    # Get categories
    if args.category:
        categories = [args.category]
    else:
        if dataset == 'ShapeNetAD':
            categories = all_shapenetad_cates
        elif dataset == 'Real3DAD':
            categories = all_real3dad_cates
        elif dataset == 'MVTec3DAD':
            categories = all_mvtec3dad_cates
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Dataset: {dataset}")
    print(f"Path: {dataset_path}")
    print(f"Categories: {len(categories)}")
    if args.use_parenet:
        print(f"Registration: PARENet (checkpoint: {args.parenet_checkpoint})")
    elif args.use_adam_9dof:
        print(f"Registration: Adam + Chamfer 9DOF")
    else:
        print(f"Registration: RANSAC + ICP (voxel_size={args.ransac_voxel_size})")

    # Evaluate all categories
    results = []
    for category in categories:
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")
        
        train_path, train_files = list_train_files(dataset_path, category, dataset)


        result = compute_pairwise_variability(train_files, train_path, args, category, dataset)

        result['category'] = category
        results.append(result)

        print(f"\n  Results:")
        print(f"    Samples: {result['n_samples']}, Pairs: {result['n_pairs']} ({result['n_successful']} successful)")
        print(f"    Fitness: {result['fitness_mean']:.4f} ± {result['fitness_std']:.4f} (min={result['fitness_min']:.4f}, max={result['fitness_max']:.4f})")
        print(f"    RMSE:    {result['rmse_mean']:.6f} ± {result['rmse_std']:.6f} (min={result['rmse_min']:.6f}, max={result['rmse_max']:.6f})")
        print(f"    Distance: {result['dist_mean']:.6f} ± {result['dist_std']:.6f} (min={result['dist_min']:.6f}, max={result['dist_max']:.6f})")


    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")


    import pandas as pd

    df = pd.DataFrame(results)
    df = df.set_index('category')

    print("\nPer-category variability:")
    print(df[['n_samples', 'fitness_mean', 'fitness_std', 'rmse_mean', 'rmse_std', 'dist_mean', 'dist_std']])

    print("\nOverall statistics across all categories:")
    print(f"  Fitness: {df['fitness_mean'].mean():.4f} ± {df['fitness_std'].mean():.4f}")
    print(f"  RMSE:    {df['rmse_mean'].mean():.6f} ± {df['rmse_std'].mean():.6f}")
    print(f"  Distance: {df['dist_mean'].mean():.6f} ± {df['dist_std'].mean():.6f}")

    # Save to CSV
    df.to_csv(args.output)
    print(f"\nResults saved to: {args.output}")


    print(f"{'='*80}")


if __name__ == '__main__':
    main()