"""
QuickFPS: GPU-accelerated Farthest Point Sampling using Bucket-based KDTree.

Implements the bucket_fps_kdtree algorithm from fpsample in CUDA.
Uses KDLineTree (bucket-based) with hierarchical pruning for fast FPS.

Expected performance: ~233-384ms for 100k→50k points (match fpsample)

Build:
    cd cuda_ops/fps_kdtree && ./build.sh

Usage:
    import torch
    from cuda_ops.fps_kdtree.fps_kdtree_quickfps_gpu import fps_kdtree_quickfps

    points = torch.randn(100000, 3, device='cuda')
    indices = fps_kdtree_quickfps(points, n_samples=3000, bucket_size=200)
"""

import torch
import sys
import os

# Add current directory to path for importing compiled module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fps_kdtree_quickfps(points, n_samples, bucket_size=200, start_idx=0):
    """
    QuickFPS: Bucket-based Farthest Point Sampling on GPU using KDTree.

    Uses KDLineTree with bounding box pruning to skip entire buckets,
    reducing complexity from O(N*M) to O(K*log(N)*M) where K << N.

    Args:
        points: (N, 3) torch.Tensor on CUDA, dtype=float32
        n_samples: Number of points to sample
        bucket_size: Max points per KDTree leaf bucket (default: 200)
        start_idx: Starting point index (default: 0)

    Returns:
        indices: (n_samples,) torch.LongTensor of selected point indices

    Example:
        >>> points = torch.randn(100000, 3, device='cuda')
        >>> indices = fps_kdtree_quickfps(points, 3000, bucket_size=200)
        >>> sampled_points = points[indices]  # (3000, 3)

    Performance:
        100k → 50k points: ~233-384ms (target)
        50k → 5k points: <50ms
    """
    # Import compiled extension
    try:
        import fps_kdtree_quickfps_cuda as fps_ext
    except ImportError:
        raise ImportError(
            "fps_kdtree_quickfps_cuda extension not built. Run:\n"
            "  cd cuda_ops/fps_kdtree && ./build.sh"
        )

    # Validate inputs
    if not points.is_cuda:
        raise ValueError("points must be on CUDA device")

    if points.dtype != torch.float32:
        points = points.float()

    if not points.is_contiguous():
        points = points.contiguous()

    if points.dim() != 2 or points.size(1) != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    N = points.size(0)
    if n_samples > N:
        raise ValueError(f"n_samples ({n_samples}) > N ({N})")

    if bucket_size <= 0 or bucket_size > N:
        bucket_size = min(200, N)

    if start_idx < 0 or start_idx >= N:
        start_idx = 0

    # Call CUDA kernel
    indices = fps_ext.fps_kdtree_quickfps(points, n_samples, bucket_size, start_idx)

    return indices


if __name__ == '__main__':
    # Quick test
    print("Testing QuickFPS CUDA...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    N = 10000
    n_samples = 1000
    bucket_size = 200

    points = torch.randn(N, 3, device='cuda')

    import time
    t0 = time.time()
    indices = fps_kdtree_quickfps(points, n_samples, bucket_size=bucket_size)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"\nQuickFPS on {N} points -> {n_samples} samples: {elapsed*1000:.1f}ms")
    print(f"  Bucket size: {bucket_size}")
    print(f"Output shape: {indices.shape}, dtype: {indices.dtype}")
    print(f"Selected indices (first 10): {indices[:10]}")

    # Verify all unique
    assert len(indices.unique()) == n_samples, "FPS should return unique indices"
    print("✓ All indices unique")
    print("✓ Test passed!")
