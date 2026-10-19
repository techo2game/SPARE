"""
Build spatial memory bank with FPFH + Points for anomaly detection.

This script builds a dual memory bank storing both FPFH features and raw point cloud patches:
1. Loads all registered training samples for a category from reg_output/
2. Extracts FIXED FPS centers from reference (sample_000.pkl)
3. For each training sample:
   - Computes FPFH features
   - Extracts patches at the same fixed centers (via KNN)
   - Stores averaged FPFH per patch
   - Stores raw point coordinates per patch
4. Builds dual memory bank:
   - FPFH matrix: (K_centers, N_samples, 33)
   - Points matrix: (K_centers, N_samples, patch_size, 3)
5. Applies coreset sampling to reduce number of centers
6. Saves the spatial memory bank with both FPFH and Points


"""
import os
import sys
import argparse
import numpy as np
import pickle
import torch
import open3d as o3d
from tqdm import tqdm
from easydict import EasyDict
import fpsample
from sklearn.neighbors import KDTree
from utils.config import cfg_from_yaml_file
from utils.misc import seed_all
from utils.dataset import all_real3dad_cates, all_shapenetad_cates, all_mvtec3dad_cates
from utils.artifacts import dataset_dir_name, registration_train_dir
from utils.fpfh_utils import compute_fpfh_features

# Add GPU FPS support
try:
    from cuda_ops.fps_kdtree.fps_kdtree_quickfps_gpu import fps_kdtree_quickfps
    GPU_FPS_AVAILABLE = True
except ImportError:
    GPU_FPS_AVAILABLE = False
    print("Warning: GPU FPS not available, using CPU fpsample")


def extract_reference_centers(ref_points, num_patches=3000):
    """
    Extract FPS centers from reference point cloud.

    Args:
        ref_points: Reference point cloud (N, 3) numpy array
        num_patches: Number of FPS centers to extract

    Returns:
        center_coords: FPS center coordinates (num_patches, 3)
        fps_idx: Indices of FPS centers in ref_points
    """
    print(f"  Extracting {num_patches} FPS centers from reference...")

    if GPU_FPS_AVAILABLE and torch.cuda.is_available():
        # GPU-accelerated QuickFPS
        ref_points_torch = torch.from_numpy(ref_points.astype(np.float32)).cuda()
        fps_idx_torch = fps_kdtree_quickfps(ref_points_torch, num_patches, bucket_size=200)
        fps_idx = fps_idx_torch.cpu().numpy()
        del ref_points_torch, fps_idx_torch
        torch.cuda.empty_cache()
    else:
        # CPU fallback
        fps_idx = fpsample.bucket_fps_kdline_sampling(ref_points, num_patches, h=5)

    center_coords = ref_points[fps_idx]  # (num_patches, 3)
    print(f"  ✓ Extracted {len(center_coords)} centers")
    return center_coords, fps_idx


def compute_patch_normals_numpy(patch_points):
    """
    Compute surface normal for a single patch using PCA (smallest eigenvector).

    Args:
        patch_points: (patch_size, 3) numpy array

    Returns:
        normal: (3,) unit normal vector
        mean_pos: (3,) patch mean position (centroid)
    """
    # Center the patch
    mean_pos = patch_points.mean(axis=0)  # (3,)
    centered = patch_points - mean_pos  # (patch_size, 3)

    # Compute covariance matrix
    cov = (centered.T @ centered) / len(patch_points)  # (3, 3)

    # Compute eigenvalues/eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Normal is eigenvector with smallest eigenvalue (first column after sorting)
        normal = eigenvectors[:, 0]  # (3,)
    except:
        # Fallback: cross product of two vectors
        if len(patch_points) > 2:
            v1 = centered[0]
            v2 = centered[len(patch_points)//2]
            normal = np.cross(v1, v2)
        else:
            normal = np.array([0., 0., 1.])  # Default

    # Normalize
    norm = np.linalg.norm(normal)
    if norm > 1e-6:
        normal = normal / norm
    else:
        normal = np.array([0., 0., 1.])  # Default for degenerate case

    return normal, mean_pos


def extract_patches_at_fixed_centers(points, fpfh, ref_centers, patch_size=128):
    """
    For each reference center, find nearest points and extract patch FPFH + Points + Normals + Means.
    Ensures spatial alignment across all training samples.

    Args:
        points: Point cloud (N, 3)
        fpfh: FPFH features (N, 33)
        ref_centers: Reference center coordinates (K, 3)
        patch_size: Number of points per patch

    Returns:
        patch_fpfh_list: List of averaged FPFH per patch (K × 33)
        patch_points_list: List of point coordinates per patch (K × patch_size × 3)
        patch_normals_list: List of patch normals (K × 3)
        patch_means_list: List of patch mean positions (K × 3)
    """
    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(points, leaf_size=30)

    # Query K nearest neighbors for each reference center
    _, knn_indices = tree.query(ref_centers, k=patch_size)  # (K, patch_size)

    patch_fpfh_list = []
    patch_points_list = []
    patch_normals_list = []
    patch_means_list = []

    for k in range(len(ref_centers)):
        indices = knn_indices[k]

        # FPFH features
        patch_fpfh = fpfh[indices]  # (patch_size, 33)
        patch_fpfh_mean = patch_fpfh.mean(axis=0)  # (33,)
        patch_fpfh_list.append(patch_fpfh_mean)

        # Point coordinates
        patch_points = points[indices]  # (patch_size, 3)
        patch_points_list.append(patch_points)

        # Compute normal and mean position for this patch
        normal, mean_pos = compute_patch_normals_numpy(patch_points)
        patch_normals_list.append(normal)
        patch_means_list.append(mean_pos)

    return patch_fpfh_list, patch_points_list, patch_normals_list, patch_means_list




def get_coreset_idx_randomp(z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
    """
    Random-projection + farthest-point coreset selection (PatchCore-style).
    Adapted from build_memory_bank.py

    Args:
        z_lib: Feature vectors (N, D) - NumPy or torch
        n: Target number of samples to keep
        eps: Random projection epsilon (preservation ratio)
        float16: Use half precision for FPS
        force_cpu: Force CPU computation

    Returns:
        coreset_idx: torch.LongTensor of selected indices (CPU)
    """
    from sklearn import random_projection

    # Ensure NumPy for sklearn
    if isinstance(z_lib, torch.Tensor):
        z_np = z_lib.detach().cpu().numpy()
    else:
        z_np = z_lib

    print(f"   Fitting random projections. Start dim = {z_np.shape}.")

    # Random projection
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_proj_np = transformer.fit_transform(z_np)
        print(f"   DONE. Transformed dim = {z_proj_np.shape}.")
    except ValueError:
        print("   Error: could not project vectors. Using original vectors.")
        z_proj_np = z_np

    # Convert to torch
    z = torch.from_numpy(z_proj_np).float()

    # Optional cast to float16
    if float16:
        z = z.half()

    # Move to CUDA if available
    if torch.cuda.is_available() and not force_cpu:
        z = z.cuda()

    # Farthest point sampling
    num_vectors = z.shape[0]
    if n > num_vectors:
        n = num_vectors

    # Initialize: pick the item with largest norm
    norms = torch.linalg.norm(z, dim=1)
    select_idx = torch.argmax(norms).item()
    coreset_idx = [select_idx]

    last_item = z[select_idx:select_idx + 1]
    min_distances = torch.linalg.norm(z - last_item, dim=1)

    for _ in tqdm(range(n - 1), desc="  Coreset FPS"):
        # Distances to last selected item
        distances = torch.linalg.norm(z - last_item, dim=1)

        # Update min distances (greedy rule)
        min_distances = torch.minimum(min_distances, distances)

        # Select farthest point
        select_idx = torch.argmax(min_distances).item()
        coreset_idx.append(select_idx)

        # Update last item
        last_item = z[select_idx:select_idx + 1]

        # Avoid reselection
        min_distances[select_idx] = 0

    # Always return CPU LongTensor
    return torch.tensor(coreset_idx, dtype=torch.long)


def coreset_sample_centers_patchbased(fpfh_matrix, points_matrix, target_num_centers):
    """
    Select centers based on patch informativeness using combined FPFH + Points features.

    Method:
    1. Flatten point coordinates and concatenate with FPFH features for each patch
    2. Flatten all patches from all centers and training samples
    3. Run coreset on all combined patches (random projection + FPS)
    4. Count how many patches from each center were selected
    5. Keep the centers that contributed the most patches to the coreset

    Args:
        fpfh_matrix: (K_centers, N_train, 33) - FPFH features
        points_matrix: (K_centers, N_train, patch_size, 3) - Point coordinates
        target_num_centers: Number of centers to keep

    Returns:
        coreset_idx: Indices of selected centers
    """
    K, N_train, D_fpfh = fpfh_matrix.shape
    _, _, patch_size, _ = points_matrix.shape

    print(f"\n  Patch-based center selection (FPFH + Points):")
    print(f"    Total centers: {K}")
    print(f"    Training samples per center: {N_train}")
    print(f"    Total patches: {K * N_train}")
    print(f"    FPFH dim: {D_fpfh}, Points dim: {patch_size * 3}, Combined: {D_fpfh + patch_size * 3}")

    # Step 1: Flatten point coordinates and concatenate with FPFH
    points_flat = points_matrix.reshape(K, N_train, -1)  # (K, N_train, patch_size*3)
    combined_matrix = np.concatenate([fpfh_matrix, points_flat], axis=2)  # (K, N_train, 33 + patch_size*3)
    print(f"    Combined features shape: {combined_matrix.shape}")

    # Step 2: Flatten all patches
    all_patches = combined_matrix.reshape(-1, combined_matrix.shape[2])  # (K * N_train, 33 + patch_size*3)
    print(f"    Flattened patches shape: {all_patches.shape}")

    # Step 3: Run coreset on ALL patches
    coreset_percentage = 0.1
    target_num_patches = int(all_patches.shape[0] * coreset_percentage)
    target_num_patches = max(target_num_patches, target_num_centers)

    print(f"    Running coreset on all patches ({coreset_percentage*100:.1f}% = {target_num_patches} patches)...")

    z_lib = torch.from_numpy(all_patches).float()
    selected_patch_indices = get_coreset_idx_randomp(
        z_lib=z_lib,
        n=target_num_patches,
        eps=0.90,
        float16=False,
        force_cpu=False
    )

    selected_patch_indices = selected_patch_indices.cpu().numpy()
    print(f"    Selected {len(selected_patch_indices)} patches via coreset")

    # Step 4: Count patches per center
    print(f"\n    Counting patches per center...")
    patch_counts = np.zeros(K, dtype=int)

    for patch_idx in selected_patch_indices:
        center_idx = patch_idx // N_train
        patch_counts[center_idx] += 1

    # Step 5: Select top centers by patch count
    print(f"    Selecting top {target_num_centers} centers by patch contribution...")

    # Sort centers by patch count (descending)
    sorted_center_indices = np.argsort(patch_counts)[::-1]
    coreset_idx = sorted_center_indices[:target_num_centers]

    # Statistics
    top_counts = patch_counts[coreset_idx]
    print(f"\n  Center selection statistics:")
    print(f"    Selected {len(coreset_idx)} centers")
    print(f"    Patch counts: min={top_counts.min()}, max={top_counts.max()}, mean={top_counts.mean():.1f}")
    print(f"    Centers with 0 patches selected: {np.sum(patch_counts == 0)}")
    print(f"    Top center contributed: {top_counts.max()} patches")

    return coreset_idx


def coreset_sample_centers(centers, target_size):
    """
    Greedy FPS on 3D center coordinates to select spatially diverse centers.

    DEPRECATED: Use coreset_sample_centers_patchbased instead for better results.

    Args:
        centers: Center coordinates (K, 3)
        target_size: Target number of centers to keep

    Returns:
        coreset_idx: Indices of selected centers (target_size,)
    """
    z = torch.from_numpy(centers).float()

    # Move to GPU if available
    if torch.cuda.is_available():
        z = z.cuda()

    num_centers = z.shape[0]
    if target_size > num_centers:
        target_size = num_centers

    # Initialize: largest norm
    norms = torch.linalg.norm(z, dim=1)
    select_idx = torch.argmax(norms).item()
    coreset_idx = [select_idx]

    last_item = z[select_idx:select_idx+1]
    min_distances = torch.linalg.norm(z - last_item, dim=1)

    for _ in tqdm(range(target_size - 1), desc="  Coreset FPS on centers"):
        distances = torch.linalg.norm(z - last_item, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        select_idx = torch.argmax(min_distances).item()
        coreset_idx.append(select_idx)
        last_item = z[select_idx:select_idx+1]
        min_distances[select_idx] = 0

    return np.array(coreset_idx)


def coreset_sample_combined(fpfh_matrix, normals_matrix, means_matrix, target_size):
    """
    Greedy FPS on combined FPFH+Normals+Means features to select most informative training samples.

    This performs coreset sampling across the training dimension (N_train),
    keeping all spatial centers but reducing the number of training samples.

    Args:
        fpfh_matrix: FPFH matrix (K, N_train, 33)
        normals_matrix: Normals matrix (K, N_train, 3)
        means_matrix: Means matrix (K, N_train, 3)
        target_size: Target number of training samples to keep

    Returns:
        coreset_idx: Indices of selected training samples (target_size,)
    """
    K, N_train, D_fpfh = fpfh_matrix.shape

    # Concatenate FPFH + Normals + Means
    combined_matrix = np.concatenate([fpfh_matrix, normals_matrix, means_matrix], axis=2)  # (K, N_train, 33+3+3=39)

    # Flatten to (N_train, K*39)
    combined_flattened = combined_matrix.transpose(1, 0, 2).reshape(N_train, -1)

    print(f"  Flattened combined features for coreset: {combined_flattened.shape}")

    z = torch.from_numpy(combined_flattened).float()

    # Move to GPU if available
    if torch.cuda.is_available():
        z = z.cuda()

    if target_size > N_train:
        target_size = N_train

    # Initialize: largest norm
    norms = torch.linalg.norm(z, dim=1)
    select_idx = torch.argmax(norms).item()
    coreset_idx = [select_idx]

    last_item = z[select_idx:select_idx+1]
    min_distances = torch.linalg.norm(z - last_item, dim=1)

    for _ in tqdm(range(target_size - 1), desc="  Coreset FPS on combined features"):
        distances = torch.linalg.norm(z - last_item, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        select_idx = torch.argmax(min_distances).item()
        coreset_idx.append(select_idx)
        last_item = z[select_idx:select_idx+1]
        min_distances[select_idx] = 0

    return np.array(coreset_idx)


def build_spatial_fpfh_memory_bank(args):
    """Build spatial memory bank with FPFH + Points for a category."""
    print("="*80)
    print("Building Spatial Memory Bank (FPFH + Points)")
    print("="*80)

    # Load config
    config = EasyDict()
    cfg_from_yaml_file(args.config, config)

    category = args.category
    dataset_type = config.dataset

    seed_all(args.seed)

    print(f"\nCategory: {category}")
    print(f"Dataset: {dataset_type}")
    print(f"Num patches: {args.num_patches}")
    print(f"Patch size: {args.patch_size}")
    print(f"Coreset percentage: {args.coreset_percentage*100:.1f}%")

    # Load all registered samples
    # Determine registration directory based on dataset type
    dataset_dir = dataset_dir_name(dataset_type)
    reg_dir = registration_train_dir(dataset_type, category)
    if reg_dir is None or not os.path.exists(reg_dir):
        raise FileNotFoundError(f"Registration directory not found: {reg_dir}\n"
                              f"Please run registration first to generate aligned samples.")

    sample_files = sorted([f for f in os.listdir(reg_dir) if f.endswith('.pkl')])
    if args.max_train_samples is not None:
        sample_files = sample_files[:args.max_train_samples]
    N_train = len(sample_files)

    print(f"\nFound {N_train} registered training samples in {reg_dir}")

    if N_train == 0:
        raise ValueError(f"No registered samples found for category {category}")

    # Load reference (sample_000.pkl)
    print("\nLoading reference (sample_000)...")
    with open(f'{reg_dir}/sample_000.pkl', 'rb') as f:
        ref_data = pickle.load(f)
    ref_points = ref_data['points']  # Already aligned
    print(f"  Reference points: {ref_points.shape}")

    # Extract FIXED reference centers
    ref_centers, _ = extract_reference_centers(ref_points, args.num_patches)

    # Build FPFH, Normals, and Means matrices
    K = args.num_patches
    fpfh_matrix = np.zeros((K, N_train, 33), dtype=np.float32)
    normals_matrix = np.zeros((K, N_train, 3), dtype=np.float32)
    means_matrix = np.zeros((K, N_train, 3), dtype=np.float32)

    print(f"\nBuilding FPFH + Normals + Means matrices:")
    print(f"  FPFH:    ({K} centers × {N_train} samples × 33 dims)")
    print(f"  Normals: ({K} centers × {N_train} samples × 3 coords)")
    print(f"  Means:   ({K} centers × {N_train} samples × 3 coords)")

    # Compute voxel size from reference
    diameter = np.linalg.norm(ref_points.max(axis=0) - ref_points.min(axis=0))
    voxel_size = diameter * 0.05
    print(f"  Voxel size: {voxel_size:.6f}")

    for idx, sample_file in enumerate(tqdm(sample_files, desc="Processing samples")):
        # Load aligned points
        with open(f'{reg_dir}/{sample_file}', 'rb') as f:
            points = pickle.load(f)['points']

        # Sample to max_points_fpfh if needed
        if len(points) > args.max_points_fpfh:
            fps_idx = fpsample.bucket_fps_kdline_sampling(points, args.max_points_fpfh, h=5)
            points = points[fps_idx]

        # Compute FPFH
        fpfh = compute_fpfh_features(points, voxel_size)

        # Extract patches at fixed centers (FPFH + Points + Normals + Means)
        patch_fpfh_list, patch_points_list, patch_normals_list, patch_means_list = extract_patches_at_fixed_centers(
            points, fpfh, ref_centers, args.patch_size
        )

        for k in range(K):
            fpfh_matrix[k, idx, :] = patch_fpfh_list[k]
            normals_matrix[k, idx, :] = patch_normals_list[k]
            means_matrix[k, idx, :] = patch_means_list[k]

    print(f"\n✓ FPFH matrix built: {fpfh_matrix.shape}")
    print(f"✓ Normals matrix built: {normals_matrix.shape}")
    print(f"✓ Means matrix built: {means_matrix.shape}")

    # Apply coreset sampling based on mode
    if args.coreset_mode == 'centers':
        # MODE 1: Coreset on CENTERS (reduce spatial locations, keep all training samples)
        target_size = int(K * args.coreset_percentage)
        print(f"\nCoreset mode: CENTERS (Spatial FPS)")
        print(f"  Percentage: {args.coreset_percentage*100:.1f}% of {K} centers = {target_size} centers")
        print(f"  Applying spatial FPS coreset selection: {K} → {target_size}")
        print(f"  Keeping ALL {N_train} training samples")

        # Use spatial FPS instead of patch-based 
        coreset_idx = coreset_sample_centers(ref_centers, target_size)
        centers_coreset = ref_centers[coreset_idx]
        fpfh_matrix_coreset = fpfh_matrix[coreset_idx]
        normals_matrix_coreset = normals_matrix[coreset_idx]
        means_matrix_coreset = means_matrix[coreset_idx]

        print(f"✓ Coreset selected: {len(coreset_idx)} centers ({len(coreset_idx)/K*100:.1f}% of total)")
        print(f"  Final FPFH shape: {fpfh_matrix_coreset.shape}")
        print(f"  Final Normals shape: {normals_matrix_coreset.shape}")
        print(f"  Final Means shape: {means_matrix_coreset.shape}")

    elif args.coreset_mode == 'samples':
        # MODE 2: Coreset on SAMPLES (keep all centers, reduce training samples)
        target_size = int(N_train * args.coreset_percentage)
        print(f"\nCoreset mode: SAMPLES (Combined FPFH+Normals+Means)")
        print(f"  Percentage: {args.coreset_percentage*100:.1f}% of {N_train} training samples = {target_size} samples")
        print(f"  Applying coreset sampling: {N_train} → {target_size}")
        print(f"  Keeping ALL {K} spatial centers")

        coreset_idx = coreset_sample_combined(fpfh_matrix, normals_matrix, means_matrix, target_size)
        centers_coreset = ref_centers
        fpfh_matrix_coreset = fpfh_matrix[:, coreset_idx, :]
        normals_matrix_coreset = normals_matrix[:, coreset_idx, :]
        means_matrix_coreset = means_matrix[:, coreset_idx, :]

        print(f"✓ Coreset selected: {len(coreset_idx)} training samples ({len(coreset_idx)/N_train*100:.1f}% of total)")
        print(f"  Final FPFH shape: {fpfh_matrix_coreset.shape}")
        print(f"  Final Normals shape: {normals_matrix_coreset.shape}")
        print(f"  Final Means shape: {means_matrix_coreset.shape}")

    else:
        # MODE 3: No coreset, keep everything
        print(f"\nNo coreset applied")
        centers_coreset = ref_centers
        fpfh_matrix_coreset = fpfh_matrix
        normals_matrix_coreset = normals_matrix
        means_matrix_coreset = means_matrix
        print(f"  Final FPFH shape: {fpfh_matrix_coreset.shape}")
        print(f"  Final Normals shape: {normals_matrix_coreset.shape}")
        print(f"  Final Means shape: {means_matrix_coreset.shape}")

    # Sanity checks
    assert not np.isnan(fpfh_matrix_coreset).any(), "NaN detected in FPFH matrix!"
    assert not np.isinf(fpfh_matrix_coreset).any(), "Inf detected in FPFH matrix!"
    assert not np.isnan(normals_matrix_coreset).any(), "NaN detected in Normals matrix!"
    assert not np.isinf(normals_matrix_coreset).any(), "Inf detected in Normals matrix!"
    assert not np.isnan(means_matrix_coreset).any(), "NaN detected in Means matrix!"
    assert not np.isinf(means_matrix_coreset).any(), "Inf detected in Means matrix!"

    # Build spatial bank dict
    spatial_bank = {
        'centers': torch.from_numpy(centers_coreset).float(),
        'fpfh_matrix': torch.from_numpy(fpfh_matrix_coreset).float(),
        'normals_matrix': torch.from_numpy(normals_matrix_coreset).float(),
        'means_matrix': torch.from_numpy(means_matrix_coreset).float(),
        'num_training': N_train,
        'category': category
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{category}_spatial_chamfer_memory_bank.pt')

    torch.save(spatial_bank, output_path)

    # Print summary
    print(f"\n{'='*80}")
    print("Spatial Memory Bank Saved (FPFH + Normals + Means)")
    print(f"{'='*80}")
    print(f"Path: {output_path}")
    print(f"Centers shape: {centers_coreset.shape}")
    print(f"FPFH matrix shape: {fpfh_matrix_coreset.shape}")
    print(f"Normals matrix shape: {normals_matrix_coreset.shape}")
    print(f"Means matrix shape: {means_matrix_coreset.shape}")
    fpfh_mem = fpfh_matrix_coreset.nbytes / 1024 / 1024
    normals_mem = normals_matrix_coreset.nbytes / 1024 / 1024
    means_mem = means_matrix_coreset.nbytes / 1024 / 1024
    total_mem = fpfh_mem + normals_mem + means_mem
    print(f"Memory: FPFH={fpfh_mem:.2f} MB, Normals={normals_mem:.2f} MB, Means={means_mem:.2f} MB, Total={total_mem:.2f} MB")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Build spatial memory bank with FPFH + Points for anomaly detection')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--category', type=str, default=None,
                        help='Category name (if not specified, runs all categories)')
    parser.add_argument('--num_patches', type=int, default=2048,
                        help='Number of FPS centers to extract from reference')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Points per patch')
    parser.add_argument('--coreset_mode', type=str, default='centers',
                        choices=['centers', 'samples', 'none'],
                        help='Coreset sampling mode: "centers" (reduce spatial centers), '
                             '"samples" (reduce training samples), "none" (no coreset)')
    parser.add_argument('--coreset_percentage', type=float, default=1.0,
                        help='Coreset percentage (0.0-1.0). For "centers" mode: %% of centers to keep. '
                             'For "samples" mode: %% of training samples to keep. Default: 0.1 (10%%)')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='spatial_memory_banks',
                        help='Output directory for spatial memory banks')
    parser.add_argument('--max_points_fpfh', type=int, default=150000,
                        help='Maximum points for FPFH computation (downsample if exceeded)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Optional cap for smoke/debug memory-bank builds')

    args = parser.parse_args()

    # If no category specified, determine categories from config
    if args.category is None:
        # Load config to determine dataset type
        config = EasyDict()
        cfg_from_yaml_file(args.config, config)

        dataset_type = config.dataset

        # Select appropriate category list based on dataset
        if dataset_type == 'Real3DAD':
            class_names = all_real3dad_cates
        elif dataset_type == 'ShapeNetAD':
            class_names = all_shapenetad_cates
        elif dataset_type == 'MVTec3DAD':
            class_names = all_mvtec3dad_cates
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Expected 'Real3DAD', 'ShapeNetAD', or 'MVTec3DAD'")

        print("\n" + "="*80)
        print(f"No category specified. Running for all {len(class_names)} categories from {dataset_type}:")
        print(f"{', '.join(class_names)}")
        print("="*80 + "\n")

        for category in class_names:
            args.category = category
            build_spatial_fpfh_memory_bank(args)
    else:
        build_spatial_fpfh_memory_bank(args)


if __name__ == '__main__':
    main()
