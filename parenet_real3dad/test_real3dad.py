"""
Test PARENet on Real3DAD test samples with visualization.

Usage:
    python parenet_real3dad/test_real3dad.py --checkpoint parenet_real3dad/output/snapshots/*/epoch-6.pth.tar
    python parenet_real3dad/test_real3dad.py --checkpoint parenet_real3dad/output/snapshots/*/epoch-6.pth.tar --visualize
"""

import _paths  # noqa: F401  (must come first)
import argparse
import glob
import os
import os.path as osp
from collections import defaultdict
import json

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree

from config import make_cfg
from model import create_model
from dataset import normalize_shape_bbox
from pareconv.utils.data import registration_collate_fn_stack_mode, precompute_neibors
from pareconv.utils.torch import release_cuda, to_cuda


def load_checkpoint(checkpoint_path, model):
    """Load model checkpoint."""
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint.get('epoch', 0)
    print(f'Loaded checkpoint from epoch {epoch}')
    return model


def create_data_dict_from_files(ref_file, src_file, cfg):
    """Create data dict from .pcd files (mimics dataset loader)."""
    # Load point clouds
    ref_pcd = o3d.io.read_point_cloud(ref_file)
    src_pcd = o3d.io.read_point_cloud(src_file)

    ref_points = np.asarray(ref_pcd.points)
    src_points = np.asarray(src_pcd.points)

    # Normalize
    ref_points = normalize_shape_bbox(ref_points)
    src_points = normalize_shape_bbox(src_points)

    # Sample to point_limit
    if cfg.test.point_limit is not None:
        if ref_points.shape[0] > cfg.test.point_limit:
            indices = np.random.permutation(ref_points.shape[0])[:cfg.test.point_limit]
            ref_points = ref_points[indices]
        if src_points.shape[0] > cfg.test.point_limit:
            indices = np.random.permutation(src_points.shape[0])[:cfg.test.point_limit]
            src_points = src_points[indices]

    # Create data dict (same format as dataset)
    data_dict = {
        'ref_points': ref_points.astype(np.float32),
        'src_points': src_points.astype(np.float32),
        'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
        'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
        'transform': np.eye(4, dtype=np.float32),  # Unknown GT
        'scene_name': f'{osp.basename(ref_file)}_{osp.basename(src_file)}',
        'ref_frame': 0,
        'src_frame': 0,
        'overlap': 0.0,
    }

    # Use collate function to preprocess (same as training/val)
    data_list = [data_dict]
    batch = registration_collate_fn_stack_mode(
        data_list,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        precompute_data=True
    )

    return batch


def register_pair(model, ref_file, src_file, cfg):
    """Register a pair of point clouds."""
    import time

    t_start = time.time()

    # Create data dict
    t0 = time.time()
    data_dict = create_data_dict_from_files(ref_file, src_file, cfg)
    t_preprocess = time.time() - t0

    # Move to GPU (same as training - use official utility)
    t0 = time.time()
    data_dict = to_cuda(data_dict)
    t_gpu = time.time() - t0

    # Compute neighbors (same as epoch_based_trainer.py does)
    t0 = time.time()
    neighbor_data = precompute_neibors(
        data_dict['points'],
        data_dict['lengths'],
        cfg.backbone.num_stages,
        cfg.backbone.num_neighbors
    )
    data_dict.update(neighbor_data)
    t_neighbors = time.time() - t0

    # Forward pass
    t0 = time.time()
    with torch.no_grad():
        output_dict = model(data_dict)
    t_model = time.time() - t0

    # Extract results
    t0 = time.time()
    estimated_transform = release_cuda(output_dict['estimated_transform'])
    ref_points = release_cuda(output_dict['ref_points'])
    src_points = release_cuda(output_dict['src_points'])
    t_extract = time.time() - t0

    t_total = time.time() - t_start

    timings = {
        'preprocess': t_preprocess,
        'gpu_transfer': t_gpu,
        'neighbors': t_neighbors,
        'model': t_model,
        'extract': t_extract,
        'total': t_total
    }

    return estimated_transform, ref_points, src_points, timings


def apply_transform(points, transform):
    """Apply 4x4 transformation to points."""
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    points_transformed = (transform @ points_h.T).T[:, :3]
    return points_transformed


def visualize_registration(ref_points, src_points, src_aligned, save_path=None):
    """Visualize registration result using Open3D."""
    # Create point clouds
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(ref_points)
    pcd_ref.paint_uniform_color([1, 0, 0])  # Red

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_points)
    pcd_src.paint_uniform_color([0, 1, 0])  # Green

    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(src_aligned)
    pcd_aligned.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize
    if save_path:
        # Save screenshot
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd_ref)
        vis.add_geometry(pcd_aligned)
        vis.update_geometry(pcd_ref)
        vis.update_geometry(pcd_aligned)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
    else:
        # Interactive visualization
        o3d.visualization.draw_geometries([pcd_ref, pcd_src, pcd_aligned],
                                         window_name="Registration Result - Red=Ref, Green=Src, Blue=Aligned",
                                         width=1280, height=720)


def test_category(model, cfg, category, templates, visualize=False, save_dir=None, max_samples=None):
    """Test all samples in a category."""
    print(f'\n{"="*60}')
    print(f'Testing category: {category}')
    print(f'{"="*60}')

    # Find test samples (Real3DAD has flat structure with defect type in filename)
    test_dir = osp.join(cfg.data.dataset_root, category, 'test')
    test_files = sorted(glob.glob(osp.join(test_dir, '*.pcd')))

    if not test_files:
        print(f'No test files found in {test_dir}')
        return [], []

    # Group by defect type (extracted from filename: XXX_defecttype.pcd)
    files_by_defect = defaultdict(list)
    for f in test_files:
        filename = osp.basename(f)
        # Parse defect type from filename (e.g., "363_bulge.pcd" -> "bulge")
        parts = filename.replace('.pcd', '').replace('_cut', '').split('_')
        if len(parts) >= 2:
            defect_type = parts[1]  # e.g., "good", "bulge", "sink"
            files_by_defect[defect_type].append(f)

    results = []
    all_timings = []

    # Get templates for this category
    category_templates = [f for cat, f in templates if cat == category]

    for defect_type, test_files in sorted(files_by_defect.items()):
        # Limit number of samples if requested
        if max_samples is not None:
            test_files = test_files[:max_samples]

        print(f'\nDefect type: {defect_type} ({len(test_files)} samples)')

        for test_file in tqdm(test_files, desc=f'{defect_type}'):
            # Register to each template and find best match
            best_chamfer = float('inf')
            best_template_file = None
            best_transform = None
            best_ref_points = None

            for template_file in category_templates:
                # Register
                try:
                    estimated_transform, ref_points, src_points, timings = register_pair(
                        model, template_file, test_file, cfg
                    )
                    all_timings.append(timings)

                    # Print inference time
                    print(f"  Registration time: {timings['total']*1000:.1f}ms (model: {timings['model']*1000:.1f}ms, preprocess: {timings['preprocess']*1000:.1f}ms, knn: {timings['neighbors']*1000:.1f}ms)")

                    # Apply transformation
                    src_aligned = apply_transform(src_points, estimated_transform)

                    # Compute chamfer distance efficiently using kd-tree
                    ref_tree = cKDTree(ref_points)
                    distances, _ = ref_tree.query(src_aligned, k=1)
                    chamfer_dist = np.mean(distances)

                    if chamfer_dist < best_chamfer:
                        best_chamfer = chamfer_dist
                        best_template_file = template_file
                        best_transform = estimated_transform
                        best_ref_points = ref_points
                        best_src_points = src_points
                        best_src_aligned = src_aligned

                except Exception as e:
                    import traceback
                    print(f'Error registering {test_file} to {template_file}:')
                    traceback.print_exc()
                    continue

            if best_template_file is None:
                print(f'Failed to register {test_file}')
                continue

            results.append({
                'category': category,
                'defect_type': defect_type,
                'test_file': osp.basename(test_file),
                'best_template': osp.basename(best_template_file),
                'chamfer_dist': float(best_chamfer),
            })

            # Visualize if requested
            if visualize:
                vis_path = None
                if save_dir:
                    os.makedirs(osp.join(save_dir, category, defect_type), exist_ok=True)
                    vis_path = osp.join(save_dir, category, defect_type,
                                       osp.basename(test_file).replace('.pcd', '.png'))

                visualize_registration(best_ref_points, best_src_points, best_src_aligned, vis_path)

    return results, all_timings


def summarize_timings(all_timings):
    """Print timing statistics."""
    if not all_timings:
        return

    print(f'\n{"="*60}')
    print('INFERENCE TIME STATISTICS')
    print(f'{"="*60}')

    # Compute averages
    avg_timings = {}
    for key in all_timings[0].keys():
        times = [t[key] for t in all_timings]
        avg_timings[key] = {
            'mean': np.mean(times) * 1000,  # Convert to ms
            'std': np.std(times) * 1000,
            'min': np.min(times) * 1000,
            'max': np.max(times) * 1000,
        }

    print(f'\nTotal registrations: {len(all_timings)}')
    print(f'\nPer-stage timings (ms):')
    print(f'  {"Stage":<15} {"Mean":<10} {"Std":<10} {"Min":<10} {"Max":<10}')
    print(f'  {"-"*55}')

    stage_names = {
        'preprocess': 'Preprocessing',
        'gpu_transfer': 'GPU Transfer',
        'neighbors': 'kNN Search',
        'model': 'Model Forward',
        'extract': 'Extract Results',
        'total': 'TOTAL'
    }

    for key in ['preprocess', 'gpu_transfer', 'neighbors', 'model', 'extract', 'total']:
        stats = avg_timings[key]
        name = stage_names[key]
        if key == 'total':
            print(f'  {"-"*55}')
        print(f'  {name:<15} {stats["mean"]:>8.2f}  {stats["std"]:>8.2f}  {stats["min"]:>8.2f}  {stats["max"]:>8.2f}')

    print(f'\n  Model inference speed: {1000.0/avg_timings["model"]["mean"]:.1f} FPS')
    print(f'  End-to-end throughput: {1000.0/avg_timings["total"]["mean"]:.1f} FPS')


def summarize_results(results):
    """Print summary statistics."""
    print(f'\n{"="*60}')
    print('REGISTRATION QUALITY METRICS')
    print(f'{"="*60}')

    if not results:
        print('No results to summarize')
        return

    # Group by category and defect type
    by_category = defaultdict(list)
    by_defect = defaultdict(list)

    for r in results:
        by_category[r['category']].append(r['chamfer_dist'])
        by_defect[f"{r['category']}_{r['defect_type']}"].append(r['chamfer_dist'])

    print(f'\nTotal samples: {len(results)}')
    print(f'\nMean Chamfer Distance: {np.mean([r["chamfer_dist"] for r in results]):.4f}')

    print('\nPer Category:')
    for cat, dists in sorted(by_category.items()):
        print(f'  {cat:15s}: {np.mean(dists):.4f} ± {np.std(dists):.4f} ({len(dists)} samples)')

    print('\nPer Defect Type:')
    for defect, dists in sorted(by_defect.items()):
        print(f'  {defect:30s}: {np.mean(dists):.4f} ± {np.std(dists):.4f} ({len(dists)} samples)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--category', default=None, help='Test specific category (default: all)')
    parser.add_argument('--visualize', action='store_true', help='Visualize registration results')
    parser.add_argument('--save_dir', default=None, help='Save visualization directory (if not set, shows interactive windows)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to test per defect type (default: all)')
    args = parser.parse_args()

    # Load config
    cfg = make_cfg()

    # Create model
    print('Creating model...')
    model = create_model(cfg).cuda()
    model = load_checkpoint(args.checkpoint, model)
    model.eval()

    # Load all training templates
    print('\nLoading training templates...')
    templates = []
    categories = sorted([d for d in os.listdir(cfg.data.dataset_root)
                        if osp.isdir(osp.join(cfg.data.dataset_root, d))])

    for category in categories:
        if args.category and category != args.category:
            continue
        train_dir = osp.join(cfg.data.dataset_root, category, 'train')
        if not osp.exists(train_dir):
            continue
        pcd_files = sorted(glob.glob(osp.join(train_dir, '*.pcd')))
        templates.extend([(category, f) for f in pcd_files])

    print(f'Loaded {len(templates)} training templates from {len(set([t[0] for t in templates]))} categories')

    # Test each category
    all_results = []
    all_timings = []
    test_categories = [args.category] if args.category else categories

    for category in test_categories:
        results, timings = test_category(model, cfg, category, templates,
                                        visualize=args.visualize,
                                        save_dir=args.save_dir if args.visualize else None,
                                        max_samples=args.max_samples)
        all_results.extend(results)
        all_timings.extend(timings)

    # Summarize
    summarize_timings(all_timings)
    summarize_results(all_results)

    # Save results
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(osp.join(args.save_dir, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nResults saved to {args.save_dir}/results.json')


if __name__ == '__main__':
    main()
