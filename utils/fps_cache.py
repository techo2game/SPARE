"""
FPS (Farthest Point Sampling) caching utility.

Simple file-based caching for static datasets.
"""
import os
import numpy as np
from pathlib import Path


def get_fps_cache_dir():
    """Get or create FPS cache directory."""
    cache_dir = Path("fps_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cached_fps_sampling(points, num_samples, h=5, prefix="", verbose=False):
    """
    Perform FPS with simple file-based caching.

    For static datasets, caches based on prefix and parameters only.

    Args:
        points: (N, 3) numpy array
        num_samples: Number of FPS samples
        h: FPS parameter
        prefix: Cache filename prefix (e.g., "bagel_ref", "test")
        verbose: Print cache hit/miss messages

    Returns:
        numpy array of FPS indices
    """
    import fpsample

    cache_dir = get_fps_cache_dir()

    # Create simple cache filename based on prefix and parameters
    if prefix:
        cache_file = cache_dir / f"{prefix}_n{num_samples}_h{h}.npy"
    else:
        cache_file = cache_dir / f"fps_n{num_samples}_h{h}.npy"

    # Try to load from cache
    if cache_file.exists():
        try:
            fps_indices = np.load(cache_file)

            # Validate cached data
            if len(fps_indices) == num_samples:
                if verbose:
                    print(f"  ✓ FPS cache hit: {cache_file.name}")
                return fps_indices
            else:
                # Invalid cache, delete it
                if verbose:
                    print(f"  ⚠ Invalid cache (wrong size), recomputing...")
                cache_file.unlink()
        except Exception as e:
            if verbose:
                print(f"  ⚠ Failed to load cache: {e}")


    if verbose:
        print(f"  ✗ FPS cache miss - computing (n={num_samples})...")

    fps_indices = fpsample.bucket_fps_kdline_sampling(points, num_samples, h=h)

    # Save to cache
    try:
        np.save(cache_file, fps_indices)
        if verbose:
            print(f"  ✓ Saved to cache: {cache_file.name}")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Failed to save cache: {e}")

    return fps_indices


def clear_fps_cache():
    """Clear all cached FPS results."""
    cache_dir = get_fps_cache_dir()

    if cache_dir.exists():
        count = 0
        for cache_file in cache_dir.glob("*.npy"):
            cache_file.unlink()
            count += 1

        print(f"Cleared {count} FPS cache files from {cache_dir}")
    else:
        print("No FPS cache directory found")


def get_cache_stats():
    """Get statistics about cached FPS results."""
    cache_dir = get_fps_cache_dir()

    if not cache_dir.exists():
        return {"count": 0, "total_size_mb": 0}

    cache_files = list(cache_dir.glob("*.npy"))
    count = len(cache_files)
    total_size = sum(f.stat().st_size for f in cache_files)

    return {
        "count": count,
        "total_size_mb": total_size / (1024 * 1024),
        "cache_dir": str(cache_dir)
    }
