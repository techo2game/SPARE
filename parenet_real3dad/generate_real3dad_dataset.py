#!/usr/bin/env python3
"""
Generate PARENet-compatible dataset from Real3DAD training samples.

Creates synthetic registration pairs by:
1. Loading complete Real3DAD templates
2. Generating reference clouds (random sampling)
3. Generating source clouds (partial masking + random sampling + transformation)
4. Computing overlap ratios
5. Saving as PyTorch tensors with metadata
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Real3DAD configuration
CATEGORIES = [
    'airplane', 'car', 'candybar', 'chicken', 'diamond', 'duck',
    'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees'
]


def normalize_shape_bbox(pc):
    """Normalize point cloud using shape_bbox mode (Real3DAD convention)."""
    pc_max = pc.max(axis=0, keepdims=True)
    pc_min = pc.min(axis=0, keepdims=True)
    shift = (pc_min + pc_max) / 2
    scale = (pc_max - pc_min).max() / 2
    pc_normalized = (pc - shift) / scale
    return pc_normalized


def random_plane_mask(points, mask_ratio, seed=None):
    """
    Generate partial point cloud by plane-cut masking.

    Simulates viewpoint occlusion by removing points on one side of a random plane.

    Args:
        points: (N, 3) array
        mask_ratio: fraction of points to REMOVE (0.3-0.7 recommended)
        seed: random seed

    Returns:
        mask: (N,) boolean array (True = keep, False = remove)
    """
    rng = np.random.default_rng(seed)

    # Random plane through centroid
    centroid = points.mean(axis=0)

    # Random normal vector (uniform on sphere)
    normal = rng.normal(size=3)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Signed distance from plane
    points_centered = points - centroid
    signed_distances = points_centered @ normal

    # Choose threshold to achieve desired mask ratio
    threshold = np.percentile(signed_distances, mask_ratio * 100)

    # Keep points on positive side (below mask_ratio percentile)
    mask = signed_distances > threshold

    return mask


def random_multi_region_mask(points, mask_ratio, num_regions=3, seed=None):
    """
    Alternative masking: Remove multiple random spherical regions.

    Args:
        points: (N, 3) array
        mask_ratio: fraction of points to REMOVE
        num_regions: number of removal regions
        seed: random seed

    Returns:
        mask: (N,) boolean array
    """
    rng = np.random.default_rng(seed)
    N = len(points)
    mask = np.ones(N, dtype=bool)

    # Compute bounding box for region sizing
    bbox_size = (points.max(axis=0) - points.min(axis=0)).max()

    for _ in range(num_regions):
        # Random center
        center = points[rng.integers(0, N)]

        # Random radius (10-30% of bbox)
        radius = rng.uniform(0.1, 0.3) * bbox_size

        # Remove points within sphere
        distances = np.linalg.norm(points - center, axis=1)
        mask &= (distances > radius)

        # Stop if we've removed enough
        if (1 - mask.sum() / N) >= mask_ratio:
            break

    return mask


def compute_overlap(ref_points, src_points, positive_radius=0.1):
    """
    Compute overlap ratio between two point clouds.

    Args:
        ref_points: (N, 3)
        src_points: (M, 3)
        positive_radius: matching threshold

    Returns:
        overlap: float in [0, 1]
    """
    # Build KDTree for reference using scipy
    from scipy.spatial import cKDTree

    ref_tree = cKDTree(ref_points)

    # Query nearest neighbors for each source point
    distances, _ = ref_tree.query(src_points, k=1)

    # Count source points within positive_radius of reference
    num_matched = np.sum(distances < positive_radius)

    overlap = num_matched / len(src_points) if len(src_points) > 0 else 0.0
    return overlap


def apply_transform(points, rotation, translation):
    """Apply rigid transformation to points."""
    return points @ rotation.T + translation


def generate_synthetic_pair(
    template_points,
    num_points_ref=30000,
    num_points_src=30000,
    mask_ratio_range=(0.3, 0.7),
    rotation_range=45.0,
    translation_range=0.2,
    matching_radius=0.1,
    seed=None
):
    """
    Generate one synthetic registration pair from a complete template.

    Args:
        template_points: (N, 3) normalized template point cloud
        num_points_ref: target number of points for reference cloud
        num_points_src: target number of points for source cloud
        mask_ratio_range: (min, max) ratio of points to remove for partial observation
        rotation_range: max rotation angle in degrees per axis
        translation_range: max translation in normalized coordinates
        matching_radius: radius for correspondence computation
        seed: random seed

    Returns:
        dict with keys:
            - ref_points: (M, 3) reference point cloud
            - src_points: (K, 3) source point cloud (partial + transformed)
            - rotation: (3, 3) ground truth rotation
            - translation: (3,) ground truth translation
            - overlap: float, overlap ratio
    """
    rng = np.random.default_rng(seed)
    N = len(template_points)

    if N > num_points_ref:
        ref_indices = rng.choice(N, num_points_ref, replace=False)
        ref_points = template_points[ref_indices]
    else:
        ref_points = template_points.copy()

    mask_ratio = rng.uniform(*mask_ratio_range)


    if rng.random() < 0.7: 
        partial_mask = random_plane_mask(template_points, mask_ratio, seed=rng.integers(0, 2**31))
    else:
        partial_mask = random_multi_region_mask(template_points, mask_ratio, num_regions=3, seed=rng.integers(0, 2**31))

    partial_points = template_points[partial_mask]

    N_partial = len(partial_points)
    if N_partial > num_points_src:
        src_indices = rng.choice(N_partial, num_points_src, replace=False)
        src_points = partial_points[src_indices]
    else:
        src_points = partial_points.copy()


    euler_angles = rng.uniform(-rotation_range, rotation_range, size=3) * np.pi / 180.0
    rotation = Rotation.from_euler('xyz', euler_angles).as_matrix().astype(np.float32)

    translation = rng.uniform(-translation_range, translation_range, size=3).astype(np.float32)

    src_points_transformed = apply_transform(src_points, rotation, translation)


    overlap = compute_overlap(ref_points, src_points_transformed, positive_radius=matching_radius)

    return {
        'ref_points': ref_points.astype(np.float32),
        'src_points': src_points_transformed.astype(np.float32),
        'rotation': rotation,
        'translation': translation,
        'overlap': float(overlap)
    }


def load_real3dad_templates(data_root, category):
    """
    Load all templates for a category.

    Args:
        data_root: path to Real3DAD root (e.g., 'data/Real3D/')
        category: category name

    Returns:
        list of (template_id, normalized_points) tuples
    """
    train_dir = Path(data_root) / category / 'train'
    templates = []

    pcd_files = sorted(train_dir.glob('*.pcd'))

    for pcd_file in pcd_files:
        template_id = pcd_file.stem

        pcd = o3d.io.read_point_cloud(str(pcd_file))
        points = np.asarray(pcd.points, dtype=np.float32)

        points_normalized = normalize_shape_bbox(points)

        templates.append((template_id, points_normalized))

    print(f"Loaded {len(templates)} templates for {category}")
    return templates


def generate_category_dataset(
    data_root,
    category,
    output_root,
    num_pairs_per_template=200,
    num_points_ref=30000,
    num_points_src=30000,
    mask_ratio_range=(0.3, 0.7),
    rotation_range=45.0,
    translation_range=0.2,
    matching_radius=0.1,
    val_template_idx=3,  # Use last template for validation
    seed=42
):
    """
    Generate synthetic dataset for one category.

    Args:
        data_root: Real3DAD data root
        category: category name
        output_root: output directory root
        num_pairs_per_template: number of synthetic pairs per template
        num_points_ref: points in reference cloud
        num_points_src: points in source cloud
        mask_ratio_range: partial masking ratio range
        rotation_range: max rotation (degrees)
        translation_range: max translation (normalized)
        matching_radius: overlap computation radius
        val_template_idx: which template index to use for validation
        seed: random seed base

    Returns:
        train_metadata: list of metadata dicts for training
        val_metadata: list of metadata dicts for validation
    """
    rng = np.random.default_rng(seed)

    # Load templates
    templates = load_real3dad_templates(data_root, category)

    train_metadata = []
    val_metadata = []

    output_dir = Path(output_root) / 'point_clouds' / category
    output_dir.mkdir(parents=True, exist_ok=True)

    for template_idx, (template_id, template_points) in enumerate(templates):
        is_val = (template_idx == val_template_idx)
        metadata_list = val_metadata if is_val else train_metadata
        split_name = 'val' if is_val else 'train'

        print(f"Generating {num_pairs_per_template} pairs for {category}/{template_id} ({split_name})")

        for pair_idx in tqdm(range(num_pairs_per_template), desc=f"{template_id}"):
            pair_seed = rng.integers(0, 2**31)
            pair_data = generate_synthetic_pair(
                template_points,
                num_points_ref=num_points_ref,
                num_points_src=num_points_src,
                mask_ratio_range=mask_ratio_range,
                rotation_range=rotation_range,
                translation_range=translation_range,
                matching_radius=matching_radius,
                seed=pair_seed
            )
            ref_filename = f"{template_id}_ref_{pair_idx:04d}.npy"
            src_filename = f"{template_id}_src_{pair_idx:04d}.npy"

            ref_path = output_dir / ref_filename
            src_path = output_dir / src_filename

            np.save(ref_path, pair_data['ref_points'])
            np.save(src_path, pair_data['src_points'])

            metadata_entry = {
                'scene_name': f"{category}_{template_id}",
                'pcd0': f"point_clouds/{category}/{ref_filename}",
                'pcd1': f"point_clouds/{category}/{src_filename}",
                'rotation': pair_data['rotation'],
                'translation': pair_data['translation'],
                'overlap': pair_data['overlap'],
                'frag_id0': pair_idx,
                'frag_id1': pair_idx,
            }

            metadata_list.append(metadata_entry)

            # Visualize first pair for inspection
            if template_idx == 0 and pair_idx == 0:

                # Create Open3D point clouds
                ref_pcd = o3d.geometry.PointCloud()
                ref_pcd.points = o3d.utility.Vector3dVector(pair_data['ref_points'])
                ref_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(pair_data['src_points'])
                src_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red

                # Add coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

                #Visualize
                # o3d.visualization.draw_geometries(
                #     [ref_pcd, src_pcd, coord_frame],
                #     window_name=f"First Pair: {category}/{template_id}",
                #     width=1024,
                #     height=768,
                #     point_show_normal=False
                # )


    return train_metadata, val_metadata


def main():
    parser = argparse.ArgumentParser(description='Generate PARENet dataset from Real3DAD')
    parser.add_argument('--data_root', type=str, default='data/Real3D',
                        help='Path to Real3DAD dataset')
    parser.add_argument('--output_root', type=str, default='parenet_real3dad/data/Real3DAD',
                        help='Output directory for generated dataset')
    parser.add_argument('--categories', nargs='+', default=CATEGORIES,
                        help='Categories to process')
    parser.add_argument('--num_pairs_per_template', type=int, default=100,
                        help='Number of synthetic pairs per template')
    parser.add_argument('--num_points_ref', type=int, default=30000,
                        help='Number of points in reference cloud')
    parser.add_argument('--num_points_src', type=int, default=30000,
                        help='Number of points in source cloud')
    parser.add_argument('--mask_ratio_min', type=float, default=0.3,
                        help='Minimum masking ratio for partial observations')
    parser.add_argument('--mask_ratio_max', type=float, default=0.7,
                        help='Maximum masking ratio for partial observations')
    parser.add_argument('--rotation_range', type=float, default=180.0,
                        help='Max rotation angle (degrees) per axis')
    parser.add_argument('--translation_range', type=float, default=0.2,
                        help='Max translation in normalized coordinates')
    parser.add_argument('--matching_radius', type=float, default=0.1,
                        help='Radius for overlap computation')
    parser.add_argument('--val_template_idx', type=int, default=3,
                        help='Template index to use for validation (0-based)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create output directories
    output_root = Path(args.output_root)
    metadata_dir = output_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_train_metadata = []
    all_val_metadata = []

    # Process each category
    for category in args.categories:
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")

        train_meta, val_meta = generate_category_dataset(
            data_root=args.data_root,
            category=category,
            output_root=args.output_root,
            num_pairs_per_template=args.num_pairs_per_template,
            num_points_ref=args.num_points_ref,
            num_points_src=args.num_points_src,
            mask_ratio_range=(args.mask_ratio_min, args.mask_ratio_max),
            rotation_range=args.rotation_range,
            translation_range=args.translation_range,
            matching_radius=args.matching_radius,
            val_template_idx=args.val_template_idx,
            seed=args.seed + hash(category) % 10000  # Category-specific seed
        )

        all_train_metadata.extend(train_meta)
        all_val_metadata.extend(val_meta)

    # Save metadata
    print(f"\nSaving metadata...")
    print(f"  Training pairs: {len(all_train_metadata)}")
    print(f"  Validation pairs: {len(all_val_metadata)}")

    with open(metadata_dir / 'train.pkl', 'wb') as f:
        pickle.dump(all_train_metadata, f)

    with open(metadata_dir / 'val.pkl', 'wb') as f:
        pickle.dump(all_val_metadata, f)

    print(f"\nDataset generation complete!")
    print(f"Output saved to: {output_root}")
    print(f"\nTo train PARENet:")
    print(f"  cd experiments/Real3DAD")
    print(f"  python trainval.py")


if __name__ == '__main__':
    main()
