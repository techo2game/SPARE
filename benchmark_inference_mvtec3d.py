"""Benchmark SPARE inference on MVTec3D-AD at a fixed point budget."""
import argparse
import csv
import json
import os
import pickle
import threading
import time

import numpy as np
import psutil
import torch
from easydict import EasyDict

from models.fusion_models import SigmoidFusion
from train_fpfh_chamfer import register_pointcloud_cached
from utils.artifacts import dataset_dir_name
from utils.config import cfg_from_yaml_file
from utils.data_loading import find_test_samples, load_reference_data_spatial_chamfer, load_test_pointcloud
from utils.fpfh_utils import compute_fpfh_features
from utils.sampling import extract_patches_gpu_fps


class PeakRamSampler:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed(stage_times, name, device):
    class Timer:
        def __enter__(self_inner):
            sync(device)
            self_inner.start = time.perf_counter()
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            sync(device)
            stage_times[name] = time.perf_counter() - self_inner.start
    return Timer()


def sample_exact(points, target_points):
    n_points = points.shape[0]
    if n_points == target_points:
        return points, None
    if n_points > target_points:
        indices = torch.randperm(n_points, device=points.device)[:target_points]
    else:
        extra = torch.randint(0, n_points, (target_points - n_points,), device=points.device)
        indices = torch.cat([torch.arange(n_points, device=points.device), extra], dim=0)
    return points[indices], indices


def load_model(checkpoint, device):
    model = SigmoidFusion().to(device)
    if checkpoint is not None:
        data = torch.load(checkpoint, map_location=device)
        state_dict = data.get('model_state_dict', data)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_mb(value):
    return value / (1024 ** 2)


def benchmark_sample(sample, model, ref_data, config, args, device):
    dataset_dir = dataset_dir_name(config.dataset)
    subdir, filename, path = sample
    stage_times = {}

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    ram_before = psutil.Process(os.getpid()).memory_info().rss
    with PeakRamSampler() as ram_sampler:
        with torch.no_grad():
            total_start = time.perf_counter()

            with timed(stage_times, 'load_s', device):
                test_data = load_test_pointcloud(path, config.dataset, subdir, is_evaluate=True)
                points_np = test_data['points_normalized']

            with timed(stage_times, 'sample_150k_s', device):
                test = torch.from_numpy(points_np).float().to(device)
                test, sampled_indices = sample_exact(test, args.target_points)

            ref_points = torch.from_numpy(ref_data['points']).float().to(device)
            cache_dir = os.path.join('reg_output', dataset_dir, args.category, 'transforms')
            cache_path = os.path.join(cache_dir, f'{subdir}_{os.path.splitext(filename)[0]}_points{args.target_points}.pkl')

            with timed(stage_times, 'registration_s', device):
                if args.skip_registration:
                    aligned = test
                else:
                    _, aligned = register_pointcloud_cached(
                        test,
                        ref_points,
                        cache_path,
                        use_cache=not args.no_registration_cache,
                    )

            with timed(stage_times, 'fpfh_s', device):
                diameter = torch.norm(aligned.max(dim=0).values - aligned.min(dim=0).values)
                fpfh = compute_fpfh_features(aligned, (diameter * 0.05).item())

            with timed(stage_times, 'patch_extraction_s', device):
                patch_points, patch_fpfh, patch_centers, patch_indices = extract_patches_gpu_fps(
                    aligned,
                    fpfh,
                    args.num_patches,
                    args.patch_size,
                )

            reference_patch_centers = ref_data['spatial_bank']['centers'].to(device)
            reference_patch_fpfh = ref_data['spatial_bank']['fpfh_matrix'].to(device)
            reference_means = ref_data['spatial_bank']['means_matrix'].to(device)
            reference_normals = ref_data['spatial_bank']['normals_matrix'].to(device)

            with timed(stage_times, 'model_forward_s', device):
                scores = model(
                    patch_points,
                    patch_fpfh,
                    patch_centers,
                    reference_patch_centers,
                    reference_means,
                    reference_normals,
                    reference_patch_fpfh,
                    k_centers=args.k_centers,
                )

            with timed(stage_times, 'postprocess_s', device):
                point_scores = torch.zeros(args.target_points, device=device)
                point_scores[patch_indices.reshape(-1)] = scores.repeat_interleave(args.patch_size)
                _ = point_scores.mean().item()

            sync(device)
            stage_times['total_s'] = time.perf_counter() - total_start

    peak_alloc = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    return {
        'category': args.category,
        'subdir': subdir,
        'file': filename,
        'input_points': int(points_np.shape[0]),
        'bench_points': args.target_points,
        'sampled_with_replacement': bool(sampled_indices is not None and points_np.shape[0] < args.target_points),
        'peak_ram_mb': tensor_mb(ram_sampler.peak),
        'ram_delta_mb': tensor_mb(ram_sampler.peak - ram_before),
        'peak_vram_allocated_mb': tensor_mb(peak_alloc),
        'peak_vram_reserved_mb': tensor_mb(peak_reserved),
        **stage_times,
    }


def summarize(rows):
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in {'input_points', 'bench_points'}]
    summary = {'num_samples': len(rows)}
    for key in numeric_keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[f'{key}_mean'] = float(values.mean())
        summary[f'{key}_p50'] = float(np.percentile(values, 50))
        summary[f'{key}_p95'] = float(np.percentile(values, 95))
    return summary


def main():
    parser = argparse.ArgumentParser(description='Benchmark SPARE inference on MVTec3D-AD')
    parser.add_argument('--dataset_config', default='configs/mvtec3d-ad/base.yaml')
    parser.add_argument('--category', default='bagel')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--target_points', type=int, default=150000)
    parser.add_argument('--num_patches', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--k_centers', type=int, default=4)
    parser.add_argument('--no_registration_cache', action='store_true')
    parser.add_argument('--skip_registration', action='store_true', help='Use identity alignment; useful when no PARENet checkpoint is available')
    parser.add_argument('--output_dir', default='artifacts/results/benchmarks')
    args = parser.parse_args()

    config = EasyDict()
    cfg_from_yaml_file(args.dataset_config, config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ref_data = load_reference_data_spatial_chamfer(args.category, dataset=dataset_dir_name(config.dataset))
    samples = find_test_samples(config.dataset_path, args.category, config.dataset)
    if not samples:
        raise FileNotFoundError(f'No test samples found for {args.category} in {config.dataset_path}')
    samples = samples[:args.warmup + args.num_samples]

    model = load_model(args.checkpoint, device)
    rows = []
    for idx, sample in enumerate(samples):
        row = benchmark_sample(sample, model, ref_data, config, args, device)
        if idx >= args.warmup:
            rows.append(row)
        status = 'warmup' if idx < args.warmup else 'measured'
        print(f"{status}: {sample[1]} total={row['total_s']:.3f}s peak_ram={row['peak_ram_mb']:.1f}MB peak_vram={row['peak_vram_allocated_mb']:.1f}MB")

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.output_dir, f'mvtec3d_{args.category}_inference_{stamp}.csv')
    json_path = os.path.join(args.output_dir, f'mvtec3d_{args.category}_inference_{stamp}.json')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'samples': rows}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Wrote {csv_path}')
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
