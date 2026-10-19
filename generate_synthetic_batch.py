"""
Batch script to generate synthetic anomalies for all categories in parallel.

Usage:
    python generate_synthetic_batch.py configs/mvtec3d-ad/base.yaml --num_workers 4
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from easydict import EasyDict
from utils.config import cfg_from_yaml_file
from utils.dataset import all_real3dad_cates, all_shapenetad_cates, all_mvtec3dad_cates


def generate_for_category(config_file, category, synth_config):
    """Generate synthetic anomalies for a single category."""
    cmd = [
        sys.executable, 'generate_synthetic_anomalies.py',
        config_file,
        '--category', category,
        '--synth_config', synth_config
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return category, True, result.stdout
    except subprocess.CalledProcessError as e:
        return category, False, e.stderr


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic anomalies for all categories in parallel')
    parser.add_argument('config', help='Dataset config file')
    parser.add_argument('--synth_config', type=str, default='configs/synthetic_anomalies_seed_based.yaml',
                        help='Synthetic anomaly config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (categories processed simultaneously)')
    parser.add_argument('--categories', '--category', nargs='+', default=None,
                        help='Specific categories to process (default: all)')
    args = parser.parse_args()

    # Load dataset config to determine which categories to use
    config = EasyDict()
    cfg_from_yaml_file(args.config, config)

    dataset_type = config.dataset

    if args.categories:
        categories = args.categories
    else:
        # Auto-detect categories based on dataset type
        if  dataset_type == 'Real3DAD':
            categories = all_real3dad_cates
        elif dataset_type == 'ShapeNetAD':
            categories = all_shapenetad_cates
        elif dataset_type == 'MVTec3DAD':
            categories = all_mvtec3dad_cates
        else:
            print(f"Unknown dataset type: {dataset_type}")
            print(f"Supported datasets: Real3DAD, ShapeNetAD, MVTec3DAD")
            return

    print("="*80)
    print(f"Generating Synthetic Anomalies - Batch Mode")
    print("="*80)
    print(f"Dataset: {dataset_type}")
    print(f"Categories: {len(categories)}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Synth config: {args.synth_config}")
    print("="*80)

    # Run generation in parallel
    results = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_category = {
            executor.submit(generate_for_category, args.config, cat, args.synth_config): cat
            for cat in categories
        }

        # Process completed tasks
        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                cat_name, success, output = future.result()
                results[cat_name] = success

                if success:
                    print(f"\n✓ {cat_name}: SUCCESS")
                    # Print last few lines of output
                    lines = output.strip().split('\n')
                    for line in lines[-3:]:
                        print(f"  {line}")
                else:
                    print(f"\n✗ {cat_name}: FAILED")
                    print(f"  Error: {output}")
            except Exception as e:
                print(f"\n✗ {category}: EXCEPTION")
                print(f"  Error: {str(e)}")
                results[category] = False

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count

    print(f"Total categories: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

    if fail_count > 0:
        print("\nFailed categories:")
        for cat, success in results.items():
            if not success:
                print(f"  - {cat}")

    print("="*80)
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()