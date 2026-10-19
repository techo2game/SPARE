"""
Batch training script for FPFH + Chamfer fusion network (config-based).

Trains fusion networks for multiple categories automatically using config files.

Usage:
    # Train all Real3DAD categories with default config
    python train_fpfh_chamfer_batch.py \
        --training_config configs/training/sigmoid_fusion.yaml \
        --dataset_config configs/real3d-ad/base.yaml

    # Train specific categories
    python train_fpfh_chamfer_batch.py \
        --training_config configs/training/sigmoid_fusion.yaml \
        --dataset_config configs/real3d-ad/base.yaml \
        --categories bagel cookie carrot

    # Train with custom device
    python train_fpfh_chamfer_batch.py \
        --training_config configs/training/sigmoid_fusion.yaml \
        --dataset_config configs/real3d-ad/base.yaml \
        --device cuda:1
"""
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from utils.dataset import all_real3dad_cates, all_shapenetad_cates, all_mvtec3dad_cates
from utils.config import cfg_from_yaml_file
from easydict import EasyDict


def get_all_categories(config_path):
    """Get all categories for the dataset specified in config."""
    config = EasyDict()
    cfg_from_yaml_file(config_path, config)

    dataset_name = config.get('dataset', 'Real3DAD')

    if dataset_name == 'Real3DAD':
        return all_real3dad_cates
    elif dataset_name == 'ShapeNetAD':
        return all_shapenetad_cates
    elif dataset_name == 'MVTec3DAD':
        return all_mvtec3dad_cates
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_single_category(training_config, dataset_config, category, args):
    """
    Train fusion network for a single category.

    Args:
        training_config: Path to training config file
        dataset_config: Path to dataset config file
        category: Category name
        args: Command-line arguments

    Returns:
        success: True if training succeeded
    """

    cmd = [
        sys.executable, 'train_fpfh_chamfer.py',
        '--training_config', training_config,
        '--dataset_config', dataset_config,
        '--category', category,
    ]

    if args.device:
        cmd.extend(['--device', args.device])

    if args.epochs is not None:
        cmd.extend(['--epochs', str(args.epochs)])

    if args.batch_size is not None:
        cmd.extend(['--batch_size', str(args.batch_size)])

    if args.visualize:
        cmd.append('--visualize')

    print("=" * 80)
    print(f"Training: {category}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {category} completed successfully in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {category} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⊘ {category} interrupted by user")
        raise


def main():
    parser = argparse.ArgumentParser(description='Batch training for FPFH + Chamfer fusion network')

    # Required configs
    parser.add_argument('--training_config', type=str, required=True,
                        help='Training config file (e.g., configs/training/sigmoid_fusion.yaml)')
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Dataset config file (e.g., configs/real3d-ad/base.yaml)')

    # Category selection
    parser.add_argument('--categories', '--category', type=str, nargs='+', default=None,
                        help='Categories to train (default: all categories in dataset)')

    # Optional overrides (override config file values)
    parser.add_argument('--device', type=str, default=None,
                        help='Device (e.g., cuda:0, cuda:1)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize during evaluation')

    # Batch execution options
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip categories that already have checkpoints')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue training other categories if one fails')

    args = parser.parse_args()

    # Verify config files exist
    if not os.path.exists(args.training_config):
        print(f"Error: Training config not found: {args.training_config}")
        sys.exit(1)
    if not os.path.exists(args.dataset_config):
        print(f"Error: Dataset config not found: {args.dataset_config}")
        sys.exit(1)

    # Load configs to display settings
    train_config = EasyDict()
    cfg_from_yaml_file(args.training_config, train_config)

    dataset_config = EasyDict()
    cfg_from_yaml_file(args.dataset_config, dataset_config)

    # Get categories to train
    if args.categories is None:
        categories = get_all_categories(args.dataset_config)
        print(f"Training all {len(categories)} categories from dataset")
    else:
        categories = args.categories
        print(f"Training {len(categories)} specified categories")

    print(f"\n{'='*80}")
    print(f"BATCH TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_config.get('dataset', 'Unknown')}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"\nConfig files:")
    print(f"  Training: {args.training_config}")
    print(f"  Dataset:  {args.dataset_config}")
    print(f"\nTraining settings:")
    print(f"  Model: {train_config.model.name}")
    print(f"  Epochs: {args.epochs if args.epochs else train_config.training.epochs}")
    print(f"  Batch size: {args.batch_size if args.batch_size else train_config.training.batch_size}")
    print(f"  Learning rate: {train_config.training.learning_rate}")
    print(f"  Loss type: {train_config.training.loss_type}")
    print(f"  Device: {args.device if args.device else 'cuda:0'}")
    print(f"\nPatch settings:")
    print(f"  Num patches (train): {train_config.patches.num_patches}")
    print(f"  Num patches (test): {train_config.patches.num_patches_test}")
    print(f"  Patch size: {train_config.patches.patch_size}")
    print(f"  K centers: {train_config.patches.k_centers}")
    print(f"\nBatch options:")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Continue on error: {args.continue_on_error}")
    print(f"{'='*80}\n")

    # Train each category
    results = {}
    start_time_total = time.time()

    try:
        for i, category in enumerate(categories, 1):
            print(f"\n{'='*80}")
            print(f"Category {i}/{len(categories)}: {category}")
            print(f"{'='*80}\n")

            # Check if already trained
            if args.skip_existing:
                log_dir = Path('logs') / category
                if log_dir.exists():
                    checkpoints = list(log_dir.glob('*/checkpoints/best.pth'))
                    if checkpoints:
                        print(f"⊘ Skipping {category} - checkpoint already exists at {checkpoints[0]}")
                        results[category] = 'skipped'
                        continue

            # Train
            success = train_single_category(
                args.training_config,
                args.dataset_config,
                category,
                args
            )
            results[category] = 'success' if success else 'failed'

            # Handle failures
            if not success and not args.continue_on_error:
                print(f"\n✗ Training failed for {category}. Stopping batch training.")
                break

    except KeyboardInterrupt:
        print("\n\n⊘ Batch training interrupted by user")

    # Summary
    elapsed_total = time.time() - start_time_total
    print("\n" + "=" * 80)
    print("BATCH TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
    print(f"\nResults:")

    success_count = sum(1 for v in results.values() if v == 'success')
    failed_count = sum(1 for v in results.values() if v == 'failed')
    skipped_count = sum(1 for v in results.values() if v == 'skipped')

    print(f"  ✓ Success: {success_count}/{len(categories)}")
    if failed_count > 0:
        print(f"  ✗ Failed:  {failed_count}/{len(categories)}")
    if skipped_count > 0:
        print(f"  ⊘ Skipped: {skipped_count}/{len(categories)}")

    print(f"\nDetailed results:")
    for category, result in results.items():
        symbol = {'success': '✓', 'failed': '✗', 'skipped': '⊘'}[result]
        print(f"  {symbol} {category}: {result}")

    # Save summary
    summary_file = f"train_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Batch Training Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training config: {args.training_config}\n")
        f.write(f"Dataset config: {args.dataset_config}\n")
        f.write(f"Total time: {elapsed_total/60:.1f} minutes\n")
        f.write(f"\nSettings:\n")
        f.write(f"  Model: {train_config.model.name}\n")
        f.write(f"  Epochs: {args.epochs if args.epochs else train_config.training.epochs}\n")
        f.write(f"  Batch size: {args.batch_size if args.batch_size else train_config.training.batch_size}\n")
        f.write(f"  Learning rate: {train_config.training.learning_rate}\n")
        f.write(f"  Loss type: {train_config.training.loss_type}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Success: {success_count}/{len(categories)}\n")
        f.write(f"  Failed:  {failed_count}/{len(categories)}\n")
        f.write(f"  Skipped: {skipped_count}/{len(categories)}\n")
        f.write(f"\nDetailed results:\n")
        for category, result in results.items():
            f.write(f"  {category}: {result}\n")

    print(f"\nSummary saved to: {summary_file}")
    print("=" * 80)

    # Exit with error code if any failed
    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
