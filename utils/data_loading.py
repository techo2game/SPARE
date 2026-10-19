"""
Data loading utilities for anomaly detection test scripts.

This module provides functions for:
- Loading reference data (templates, FPFH, patches, memory banks)
- Finding and loading test samples
- Point cloud normalization
"""
import os
import sys
import numpy as np
import open3d as o3d
import torch
import pickle
from PIL import Image
from utils.artifacts import reference_sample_path, spatial_chamfer_bank_path

def normalize_shape_bbox(pc):
    """
    Normalize point cloud using shape_bbox mode.

    Centers the point cloud at the bounding box center and scales by
    the maximum bounding box dimension.

    Args:
        pc: Point cloud array (N, 3)

    Returns:
        Normalized point cloud array (N, 3)
    """
    pc_max = pc.max(axis=0, keepdims=True)
    pc_min = pc.min(axis=0, keepdims=True)
    shift = (pc_min + pc_max) / 2
    scale = (pc_max - pc_min).max() / 2
    if scale <= 0:
        scale = 1.0
    pc_normalized = (pc - shift) / scale
    return pc_normalized

def list_train_files(dataset_path, category, dataset_type):
    """
    List the training point clouds of a category.

    MVTec3D-AD stores organized point clouds as train/good/xyz/*.tiff;
    Real3DAD and ShapeNetAD store train/*.pcd.

    Returns:
        (train_path, sorted list of file names)
    """
    if dataset_type == 'MVTec3DAD':
        train_path = os.path.join(dataset_path, category, 'train', 'good', 'xyz')
        ext = '.tiff'
    else:
        train_path = os.path.join(dataset_path, category, 'train')
        ext = '.pcd'
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    train_files = sorted(f for f in os.listdir(train_path) if f.endswith(ext))
    if not train_files:
        raise FileNotFoundError(f"No {ext} training files in {train_path}")
    return train_path, train_files


def load_train_pointcloud(file_path):
    """Load a training point cloud as (N, 3) float32 (MVTec TIFF: valid points only)."""
    if file_path.endswith(('.tiff', '.tif')):
        import tifffile
        organized_pc = tifffile.imread(file_path)
        valid_mask = organized_pc[:, :, 2] != 0
        return organized_pc[valid_mask].astype(np.float32)
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points, dtype=np.float32)

def load_reference_data_spatial_chamfer(category, dataset='real3d-ad'):
    """
    Load reference data from pickle and memory bank.

    Args:
        category: Category name
        dataset: Dataset name (e.g., 'real3d-ad', 'shapenet-ad', 'mvtec3d-ad')

    Returns:
        dict with keys:
            - points: reference point cloud
            - spatial_bank: spatial memory bank with 'centers', 'fpfh_matrix', 'points_matrix'
    """
    ref_path = reference_sample_path(dataset, category)
    if ref_path is None or not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Reference sample not found for {dataset}/{category}. "
            "Run compute_train_variability.py --use_parenet --save_transforms first."
        )
    with open(ref_path, 'rb') as f:
        ref_data = pickle.load(f)

    # Load spatial chamfer memory bank
    spatial_bank_path = spatial_chamfer_bank_path(category)
    if spatial_bank_path is None:
        raise FileNotFoundError(
            f"Spatial chamfer memory bank not found for {category}. "
            "Run build_spatial_fpfh_memory_bank.py or place it in spatial_memory_banks/."
        )
    spatial_bank = torch.load(spatial_bank_path, weights_only=False)
    ref_data['spatial_bank'] = spatial_bank

    return ref_data

def find_test_samples(dataset_path, category, dataset_type, test_file=None):
    """
    Find test samples to process.

    Args:
        dataset_path: Root path to dataset
        category: Category name
        dataset_type: 'Real3DAD'
        test_file: Specific test file to process, or None for all files

    Returns:
        list of tuples: [(subdir, filename, full_path), ...]
    """
    test_path = os.path.join(dataset_path, category, 'test')
    test_samples = []

    if test_file is not None:
        # Process single file
        for subdir in os.listdir(test_path):
            subdir_path = os.path.join(test_path, subdir)
            if os.path.isdir(subdir_path):
                # Check directly in subdir
                candidate = os.path.join(subdir_path, test_file)
                if os.path.exists(candidate):
                    test_samples.append((subdir, test_file, candidate))
                    break
                # Check in xyz subdirectory (MVTec3D-AD format)
                candidate_xyz = os.path.join(subdir_path, 'xyz', test_file)
                if os.path.exists(candidate_xyz):
                    test_samples.append((subdir, test_file, candidate_xyz))
                    break

        if len(test_samples) == 0:
            candidate = os.path.join(test_path, test_file)
            if os.path.exists(candidate):
                test_samples.append(('unknown', test_file, candidate))
            else:
                raise FileNotFoundError(f"Test file not found: {test_file}")
    else:

        if dataset_type == 'MVTec3DAD':
            for subdir in sorted(os.listdir(test_path), key=lambda x: (x == 'good', x)):
                xyz_dir = os.path.join(test_path, subdir, 'xyz')
                if not os.path.isdir(xyz_dir):
                    continue
                for f in sorted(os.listdir(xyz_dir)):
                    if f.endswith('.tiff'):
                        test_samples.append((subdir, f, os.path.join(xyz_dir, f)))
            return test_samples

        # Real3D-AD and ShapeNetAD: test/*.pcd (directly in test directory)
        # Sort with "good" files at the end for easier visualization of bad samples
        files = sorted(os.listdir(test_path), key=lambda x: ('good' in x.lower(), x))
        for f in files:
            if f.endswith('.pcd'):
                file_path = os.path.join(test_path, f)
                if os.path.isfile(file_path):
                    test_samples.append(('test', f, file_path))

    return test_samples

def load_test_pointcloud(file_path, dataset_type, subdir, is_evaluate):
    """
    Load test point cloud from file.

    Args:
        file_path: Path to test file (.tiff or .pcd)
        dataset_type: 'Real3DAD', or 'ShapeNetAD'
        subdir: Subdirectory name (e.g., 'good', 'crack', etc.)
        is_evaluate: Whether in evaluation mode

    Returns:
        dict with keys:
            - points_raw: raw point cloud
            - points_normalized: normalized points
            - gt_mask: ground truth labels (for Real3D-AD) or None
    """
    if file_path.endswith(('.tiff', '.tif')) or dataset_type == 'MVTec3DAD':
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("MVTec3D-AD TIFF loading requires tifffile") from exc

        organized_pc = tifffile.imread(file_path)
        valid_mask_2d = organized_pc[:, :, 2] != 0
        pixel_indices = np.argwhere(valid_mask_2d)
        original_shape = organized_pc.shape[:2]
        test_points_raw = organized_pc[valid_mask_2d].astype(np.float32)

        gt_mask = np.zeros(len(test_points_raw), dtype=np.float32)
        if is_evaluate and subdir != 'good':
            gt_path = file_path.replace(f'{os.sep}xyz{os.sep}', f'{os.sep}gt{os.sep}')
            gt_path = os.path.splitext(gt_path)[0] + '.png'
            if os.path.exists(gt_path):
                gt_img = np.array(Image.open(gt_path).convert('L'))
                gt_mask = (gt_img[pixel_indices[:, 0], pixel_indices[:, 1]] > 0).astype(np.float32)

        rgb_path = file_path.replace(f'{os.sep}xyz{os.sep}', f'{os.sep}rgb{os.sep}')
        rgb_path = os.path.splitext(rgb_path)[0] + '.png'
        colors_raw = None
        if os.path.exists(rgb_path):
            rgb_img = np.array(Image.open(rgb_path).convert('RGB'))
            colors_raw = rgb_img[pixel_indices[:, 0], pixel_indices[:, 1]]

        test_points_normalized = normalize_shape_bbox(test_points_raw)
        return {
            'points_raw': test_points_raw,
            'points_normalized': test_points_normalized,
            'gt_mask': gt_mask,
            'pixel_indices': pixel_indices,
            'original_shape': original_shape,
            'is_mvtec': True,
            'colors_raw': colors_raw,
        }

    # Determine GT directory name based on dataset type
    # ShapeNetAD uses 'GT' (uppercase), Real3DAD uses 'gt' (lowercase)
    gt_dir_name = 'GT' if dataset_type == 'ShapeNetAD' else 'gt'


    # Load GT mask for Real3D-AD and ShapeNetAD
    # Check if it's a normal sample by looking at the filename
    is_good_sample = 'good' in os.path.basename(file_path).lower() or 'positive' in os.path.basename(file_path).lower()

    gt_file = file_path.replace('/test/', f'/{gt_dir_name}/').replace('.pcd', '.txt')
    if os.path.exists(gt_file) and not is_good_sample:
        # Anomalous sample with GT file
        # ShapeNetAD uses comma delimiter, Real3DAD uses whitespace
        if dataset_type == 'ShapeNetAD':
            pointcloud_mask = np.loadtxt(gt_file, dtype=np.float32, delimiter=',')
        else:
            pointcloud_mask = np.loadtxt(gt_file, dtype=np.float32)
        gt_mask = pointcloud_mask[:, 3]  # Extract label column
        test_points_raw = pointcloud_mask[:, :3]  # x, y, z
    else:
        # Load point cloud normally
        test_pcd_raw = o3d.io.read_point_cloud(file_path)
        test_points_raw = np.array(test_pcd_raw.points, dtype=np.float32)
        gt_mask = np.zeros(len(test_points_raw))



    pixel_indices = None
    original_shape = None

    test_points_normalized = normalize_shape_bbox(test_points_raw)

    return {
        'points_raw': test_points_raw,
        'points_normalized': test_points_normalized,
        'gt_mask': gt_mask,
        'pixel_indices': pixel_indices,
        'original_shape': original_shape,
        'is_mvtec': False,
        'colors_raw': None,
    }
