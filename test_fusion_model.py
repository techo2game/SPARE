"""
Test Fusion model - Evaluate I-AUROC and P-AUROC metrics
"""
import os
import sys
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.data_loading import (
    load_reference_data_spatial_chamfer,
    find_test_samples,
    load_test_pointcloud
)
from utils.dataset import all_real3dad_cates, all_shapenetad_cates, all_mvtec3dad_cates
from utils.fpfh_utils import compute_fpfh_features
from utils.sampling import extract_patches_gpu_fps
from models.fusion_models import SigmoidFusion
from models.fusion_models_optimized import SigmoidFusionOptimized
from registration.methods.parenet_torch import parenet_torch

# Visualization (optional)
try:
    from registration.pyvista_visualizer import visualize_anomaly_scores
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def apply_transformation_torch(points, transform):
    """Apply 4x4 transformation matrix to torch tensor points on GPU."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    points_transformed = (points @ R.T) + t
    return points_transformed

# FPS methods
try:
    from cuda_ops.fps_kdtree.fps_kdtree_quickfps_gpu import fps_kdtree_quickfps
    HAS_QUICKFPS = True
except ImportError:
    print("Warning: QuickFPS not available")
    HAS_QUICKFPS = False

try:
    from torch_cluster import fps as torch_cluster_fps
    HAS_TORCH_CLUSTER = True
except ImportError:
    print("Warning: torch_cluster not available")
    HAS_TORCH_CLUSTER = False

from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster


def voxel_downsample_indices(points, target_points):
    """
    Voxel downsampling that returns indices to original points.
    Adaptively finds voxel size to achieve ~target point count.

    Args:
        points: (N, 3) torch.Tensor on GPU
        target_points: Target number of points

    Returns:
        indices: (M,) indices of selected points
    """
    # Estimate initial voxel size from bounding box
    extent = points.max(dim=0).values - points.min(dim=0).values
    volume = extent.prod().item()
    voxel_size = (volume / target_points) ** (1/3)

    # Binary search for good voxel size
    low, high = voxel_size * 0.1, voxel_size * 5.0

    for _ in range(10):
        mid = (low + high) / 2
        cluster = voxel_grid(points, size=mid)
        _, indices = consecutive_cluster(cluster)
        n_unique = indices.shape[0]

        if n_unique < target_points * 0.9:
            high = mid
        elif n_unique > target_points * 1.1:
            low = mid
        else:
            break

    return indices


def load_model_and_config(checkpoint_path, device, use_optimized=False):
    """Load trained model and training config from checkpoint directory."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_class = checkpoint.get('model_class', 'SigmoidFusion')

    # Choose between original and optimized versions
    if use_optimized and model_class == 'SigmoidFusion':
        model_classes = {
            'SigmoidFusion': SigmoidFusionOptimized
        }
        print(f"  Using optimized {model_class}")
    else:
        model_classes = {
            'SigmoidFusion': SigmoidFusion,
        }
        print(f"  Using original {model_class}")

    model = model_classes[model_class]().to(device)

    # Load weights (works for both original and optimized)
    if use_optimized and model_class == 'SigmoidFusion':
        # Map original weights to optimized model
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # Rename geometry_fusion.weight_predictor → geometry_fusion_mlp
            if k.startswith('geometry_fusion.weight_predictor.'):
                new_key = k.replace('geometry_fusion.weight_predictor.', 'geometry_fusion_mlp.')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # Load training config from the log directory
    checkpoint_dir = Path(checkpoint_path).parent
    log_dir = checkpoint_dir.parent

    training_config_path = log_dir / 'training_config.yaml'
    if not training_config_path.exists():
        # Fallback: try dataset_config.yaml (older format)
        training_config_path = log_dir / 'dataset_config.yaml'

    if training_config_path.exists():
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
    else:
        print(f"Warning: Training config not found at {training_config_path}")
        training_config = {}

    return model, training_config



class TestDataset(Dataset):
    """PyTorch Dataset for test samples."""

    def __init__(self, test_samples, dataset_config):
        """
        Args:
            test_samples: List of (subdir, filename, file_path) tuples
            dataset_config: Dataset configuration dict
        """
        self.test_samples = test_samples
        self.dataset_config = dataset_config
        self.dataset_type = dataset_config.get('dataset', 'Real3DAD')

    def __len__(self):
        return len(self.test_samples)

    def __getitem__(self, idx):
        subdir, filename, file_path = self.test_samples[idx]

        # Load test point cloud with ground truth (this runs in worker process)
        test_data = load_test_pointcloud(
            file_path,
            self.dataset_type,
            subdir,
            is_evaluate=True  # Load ground truth
        )

        # Convert to tensors here (in worker process, parallel)
        points_tensor = torch.from_numpy(test_data['points_normalized']).float()
        gt_mask_tensor = torch.from_numpy(test_data['gt_mask']).long()

        return {
            'points': points_tensor,  # Already a tensor
            'gt_mask': gt_mask_tensor,  # Already a tensor
            'filename': filename,
            'subdir': subdir
        }


def collate_fn(batch):
    """Custom collate function to handle variable-sized point clouds."""
    # Return list of dicts (no batching, process one at a time)
    return batch[0] if len(batch) == 1 else batch


def find_latest_checkpoint(category, epoch=None):
    """
    Find the latest checkpoint for a given category.

    Args:
        category: Category name (e.g., 'airplane', 'bagel')
        epoch: Specific epoch number (e.g., 5 for epoch_005.pth), or None for latest

    Returns:
        Path to checkpoint file
    """
    logs_dir = Path('logs') / category

    if not logs_dir.exists():
        raise FileNotFoundError(f"No logs found for category '{category}' at {logs_dir}")

    # Find all training runs (sorted by timestamp, most recent first)
    training_runs = sorted([d for d in logs_dir.iterdir() if d.is_dir()], reverse=True)

    if not training_runs:
        raise FileNotFoundError(f"No training runs found in {logs_dir}")

    # Use the most recent training run
    latest_run = training_runs[0]
    checkpoints_dir = latest_run / 'checkpoints'

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {latest_run}")

    if epoch is not None:
        # Load specific epoch
        checkpoint_path = checkpoints_dir / f'epoch_{epoch:03d}.pth'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Find latest epoch checkpoint
        epoch_files = sorted(checkpoints_dir.glob('epoch_*.pth'))
        if not epoch_files:
            raise FileNotFoundError(f"No epoch checkpoints found in {checkpoints_dir}")
        checkpoint_path = epoch_files[-1]  # Latest epoch

    return checkpoint_path


def evaluate_category(category, device, dataset_config, checkpoint_path=None, epoch=None, num_samples=None, testing_config=None, num_workers=None, visualize=False):
    """Evaluate a single category and return metrics."""

    # Auto-detect checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(category, epoch)
        print(f"  Auto-detected: {checkpoint_path}")

    # Load model and training config
    model, training_config = load_model_and_config(checkpoint_path, device, use_optimized=False)

    # Use testing config if provided, otherwise use training config parameters
    if testing_config is not None:
        patches_config = testing_config.get('patches', {})
        eval_config = testing_config.get('evaluation', {})
        print(f"  Using testing config parameters")
    else:
        patches_config = training_config.get('patches', {})
        eval_config = training_config.get('evaluation', {})
        print(f"  Using training config parameters")

    num_patches = patches_config.get('num_patches_test', patches_config.get('num_patches', 3000))
    patch_size = patches_config.get('patch_size', 128)
    k_centers = patches_config.get('k_centers', 10)
    max_points = patches_config.get('max_points', 150000)
    batch_size = eval_config.get('batch_size', 1)

    # Override num_workers from config if not provided via command-line
    if num_workers is None:
        num_workers = eval_config.get('num_workers', 4)

    print(f"  Parameters: num_patches={num_patches}, patch_size={patch_size}, k_centers={k_centers}, max_points={max_points}")
    print(f"  Evaluation: batch_size={batch_size}, num_workers={num_workers}")

    # Create namespace with all parameters
    class EvalArgs:
        pass
    eval_args = EvalArgs()
    eval_args.num_patches = num_patches
    eval_args.patch_size = patch_size
    eval_args.k_centers = k_centers
    eval_args.max_points = max_points


    dataset_name = dataset_config.get('dataset', 'real3d-ad').lower().replace('ad', '-ad')
    ref_data = load_reference_data_spatial_chamfer(category, dataset_name)


    if isinstance(ref_data['points'], np.ndarray):
        ref_data['points'] = torch.from_numpy(ref_data['points']).float().to(device)
    else:
        ref_data['points'] = ref_data['points'].to(device)

    ref_data['spatial_bank']['centers'] = ref_data['spatial_bank']['centers'].to(device)
    ref_data['spatial_bank']['fpfh_matrix'] = ref_data['spatial_bank']['fpfh_matrix'].to(device)
    ref_data['spatial_bank']['means_matrix'] = ref_data['spatial_bank']['means_matrix'].to(device)
    ref_data['spatial_bank']['normals_matrix'] = ref_data['spatial_bank']['normals_matrix'].to(device)

    # Find test samples
    test_samples = find_test_samples(
        dataset_config['dataset_path'],
        category,
        dataset_config['dataset']
    )

    if num_samples is not None:
        test_samples = test_samples[:num_samples]


    test_dataset = TestDataset(test_samples, dataset_config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=2 if num_workers > 0 else None,  
        persistent_workers=True if num_workers > 0 else False  
    )

    print(f"  Evaluating {len(test_samples)} samples with {num_workers} workers (prefetch_factor=2)...")


    all_results = []
    image_scores = []
    image_labels = []


    with torch.no_grad(), tqdm(total=len(test_loader), desc=f"  {category}", ncols=100) as pbar:
        for batch in test_loader:

            test_full = batch['points'].to(device, non_blocking=True) 
            gt_mask_tensor = batch['gt_mask']

            if test_full.shape[0] > eval_args.max_points:
                n_before = test_full.shape[0]
                indices = voxel_downsample_indices(test_full, eval_args.max_points)
                test = test_full[indices]
                gt_mask = gt_mask_tensor[indices.cpu()].cpu().numpy()
            else:
                test = test_full
                gt_mask = gt_mask_tensor.cpu().numpy()

            # Registration using PARENET
            transform = parenet_torch(test, ref_data['points'])
            test_aligned = apply_transformation_torch(test, transform)

            # FPFH computation
            diameter = torch.norm(test_aligned.max(dim=0).values - test_aligned.min(dim=0).values)
            voxel_size = diameter * 0.05
            test_fpfh = compute_fpfh_features(test_aligned, voxel_size.item())

            # Patch extraction
            patch_points_gpu, patch_fpfh_gpu, patch_centers_gpu, patch_indices = extract_patches_gpu_fps(
                test_aligned, test_fpfh, eval_args.num_patches, eval_args.patch_size
            )

            anomaly_scores = model(
                patch_points_gpu,                          # (num_patches, patch_size, 3)
                patch_fpfh_gpu,                           # (num_patches, 33)
                patch_centers_gpu,                        # (num_patches, 3)
                ref_data['spatial_bank']['centers'],      # (R, 3)
                ref_data['spatial_bank']['means_matrix'], # (R, N, 3)
                ref_data['spatial_bank']['normals_matrix'], # (R, N, 3)
                ref_data['spatial_bank']['fpfh_matrix'],  # (R, N, 33)
                k_centers=eval_args.k_centers
            )  # (num_patches,)

            point_scores = torch.zeros(test_aligned.shape[0], device=device)
            for i, indices in enumerate(patch_indices):
                point_scores[indices] = torch.max(point_scores[indices], anomaly_scores[i])

            point_scores_cpu = point_scores.cpu().numpy()

            if visualize and HAS_PYVISTA:
                subdir = batch['subdir']
                filename = batch['filename']
                image_label = int((gt_mask > 0).any())
                visualize_anomaly_scores(
                    points=test.cpu().numpy(),
                    scores=point_scores_cpu,
                    gt_mask=gt_mask,
                    title=f"{subdir}/{filename} (Label: {'Anomaly' if image_label == 1 else 'Normal'})",
                    show=True
                )

            image_score = float(point_scores_cpu.max())

            image_label = int((gt_mask > 0).any())

            p_auroc = None
            if len(np.unique(gt_mask)) > 1:
                p_auroc = roc_auc_score(gt_mask, point_scores_cpu)

            result = {
                'image_score': image_score,
                'image_label': image_label,
                'point_scores': point_scores_cpu,
                'gt_mask': gt_mask,
                'p_auroc': p_auroc
            }

            all_results.append(result)
            image_scores.append(result['image_score'])
            image_labels.append(result['image_label'])

            pbar.update(1)


    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)

    p_aurocs = [r['p_auroc'] for r in all_results if r['p_auroc'] is not None]

    i_auroc = None
    if len(np.unique(image_labels)) > 1:
        i_auroc = roc_auc_score(image_labels, image_scores)

    p_auroc_mean = None
    p_auroc_std = None
    if len(p_aurocs) > 0:
        p_auroc_mean = np.mean(p_aurocs)
        p_auroc_std = np.std(p_aurocs)

    num_normal = int((image_labels == 0).sum())
    num_anomalous = int((image_labels == 1).sum())

    return {
        'category': category,
        'i_auroc': i_auroc,
        'p_auroc_mean': p_auroc_mean,
        'p_auroc_std': p_auroc_std,
        'num_normal': num_normal,
        'num_anomalous': num_anomalous,
        'num_samples': len(test_samples),
        'num_p_auroc_samples': len(p_aurocs)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate fusion model with I-AUROC and P-AUROC metrics')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (optional, auto-detected if not provided)')
    parser.add_argument('--category', type=str, nargs='+', default=None, help='One or more categories (default: all)')
    parser.add_argument('--all', action='store_true', help='Evaluate all Real3D-AD categories')
    parser.add_argument('--epoch', type=int, default=None, help='Specific epoch to load (default: latest epoch)')
    parser.add_argument('--dataset_config', type=str, default='configs/real3d-ad/base.yaml', help='Dataset config')
    parser.add_argument('--testing_config', type=str, default=None, help='Testing config to override training parameters (e.g., configs/testing/sigmoid_fusion.yaml)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of test samples to evaluate per category (default: all)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of DataLoader workers (default: from config or 4)')
    parser.add_argument('--viz', action='store_true', help='Visualize each test sample with anomaly scores')

    args = parser.parse_args()

    # Check visualization availability
    if args.viz and not HAS_PYVISTA:
        print("Warning: --viz requested but pyvista_visualizer not available, disabling visualization")
        args.viz = False

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset config
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)

    # Load testing config if provided
    testing_config = None
    if args.testing_config is not None:
        print(f"\nLoading testing config from {args.testing_config}")
        with open(args.testing_config, 'r') as f:
            testing_config = yaml.safe_load(f)

    if args.category is not None:
        categories = args.category
    else:
        dataset_type = dataset_config.get('dataset', 'Real3DAD')
        if dataset_type == 'Real3DAD':
            categories = all_real3dad_cates
        elif dataset_type == 'ShapeNetAD':
            categories = all_shapenetad_cates
        elif dataset_type == 'MVTec3DAD':
            categories = all_mvtec3dad_cates
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"\nEvaluating all {len(categories)} {dataset_type} categories")

    # Evaluate each category
    all_category_results = []

    failed_categories = []
    for i, category in enumerate(categories, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(categories)}] Evaluating category: {category}")
        print(f"{'='*80}")

        try:
            result = evaluate_category(
                category, device, dataset_config,
                checkpoint_path=args.checkpoint,
                epoch=args.epoch,
                num_samples=args.num_samples,
                testing_config=testing_config,
                num_workers=args.num_workers,
                visualize=args.viz
            )
            all_category_results.append(result)


            print(f"\n  Results for {category}:")
            if result['i_auroc'] is not None:
                print(f"    I-AUROC:  {result['i_auroc']:.4f}")
            else:
                print(f"    I-AUROC:  N/A")

            if result['p_auroc_mean'] is not None:
                print(f"    P-AUROC:  {result['p_auroc_mean']:.4f} ± {result['p_auroc_std']:.4f}")
            else:
                print(f"    P-AUROC:  N/A")

            print(f"    Samples:  {result['num_normal']} normal, {result['num_anomalous']} anomalous")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: Failed to evaluate {category}: {e}")
            failed_categories.append(category)
            continue

    # Print summary if evaluating multiple categories
    if len(categories) > 1:
        print("\n" + "="*80)
        print("SUMMARY ACROSS ALL CATEGORIES")
        print("="*80)

        # Table header
        print(f"\n{'Category':<15} {'I-AUROC':<12} {'P-AUROC':<20} {'Samples':<15}")
        print("-"*80)

        # Per-category results
        i_aurocs = []
        p_aurocs = []

        for result in all_category_results:
            cat = result['category']
            i_auc = f"{result['i_auroc']:.4f}" if result['i_auroc'] is not None else "N/A"

            if result['p_auroc_mean'] is not None:
                p_auc = f"{result['p_auroc_mean']:.4f} ± {result['p_auroc_std']:.4f}"
                p_aurocs.append(result['p_auroc_mean'])
            else:
                p_auc = "N/A"

            samples = f"{result['num_normal']}N + {result['num_anomalous']}A"

            print(f"{cat:<15} {i_auc:<12} {p_auc:<20} {samples:<15}")

            if result['i_auroc'] is not None:
                i_aurocs.append(result['i_auroc'])

        # Average metrics
        print("-"*80)
        if i_aurocs:
            avg_i_auroc = np.mean(i_aurocs)
            print(f"{'AVERAGE':<15} {avg_i_auroc:.4f}")

        if p_aurocs:
            avg_p_auroc = np.mean(p_aurocs)
            print(f"{'(P-AUROC avg)':<15} {'':<12} {avg_p_auroc:.4f}")

        print("="*80)

    if failed_categories:
        # Averages above exclude these categories: do not report them as complete results.
        print(f"\nFAILED categories ({len(failed_categories)}/{len(categories)}): {', '.join(failed_categories)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
