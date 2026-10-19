"""
Dataset and data loader for Real3DAD registration training.

On-the-fly pair generation from raw .pcd files.
Based on ModelNet dataset structure.
"""

import os
import os.path as osp
import glob
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data
import open3d as o3d

from pareconv.utils.pointcloud import (
    random_sample_transform,
    apply_transform,
    inverse_transform,
    get_transform_from_rotation_translation,
)
from pareconv.utils.registration import compute_overlap
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    build_dataloader_stack_mode,
)
from pareconv.transforms.functional import (
    random_shuffle_points,
    random_sample_points,
    random_crop_point_cloud_with_plane,
    random_jitter_points,
)


def normalize_shape_bbox(pc):
    """Normalize point cloud using shape_bbox mode (Real3DAD convention)."""
    pc_max = pc.max(axis=0, keepdims=True)
    pc_min = pc.min(axis=0, keepdims=True)
    shift = (pc_min + pc_max) / 2
    scale = (pc_max - pc_min).max() / 2
    pc_normalized = (pc - shift) / scale
    return pc_normalized


class Real3DADPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1.0,
        augmentation_crop=False,
        point_keep_ratio=0.7,
        rotation_range=180.0,
        translation_range=0.2,
        matching_radius=None,
        **kwargs
    ):
        super(Real3DADPairDataset, self).__init__()
        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.augmentation_crop = augmentation_crop
        self.point_keep_ratio = point_keep_ratio
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.matching_radius = matching_radius

        # Find all training .pcd files
        self.template_files = []
        categories = sorted([d for d in os.listdir(dataset_root)
                           if osp.isdir(osp.join(dataset_root, d))])

        for category in categories:
            train_dir = osp.join(dataset_root, category, 'train')
            if not osp.exists(train_dir):
                continue
            pcd_files = sorted(glob.glob(osp.join(train_dir, '*.pcd')))
            self.template_files.extend([(category, f) for f in pcd_files])

        # Split train/val (80/20)
        random.seed(7351)
        random.shuffle(self.template_files)
        split_idx = int(0.8 * len(self.template_files))
        if subset == 'train':
            self.template_files = self.template_files[:split_idx]
        else:
            self.template_files = self.template_files[split_idx:]

        print(f'{subset}: {len(self.template_files)} templates')

        # Cache loaded point clouds for validation (avoid disk I/O bottleneck)
        self.cached_templates = {}
        if subset == 'val':
            print(f'Preloading {len(self.template_files)} validation templates...')
            for category, pcd_file in self.template_files:
                raw_points = self._load_and_normalize_pcd(pcd_file)
                self.cached_templates[pcd_file] = raw_points
            print(f'Validation templates cached!')

    def __len__(self):
        # Each template generates multiple pairs per epoch
        if self.subset == 'train':
            return len(self.template_files) * 100  # 100 pairs per template
        else:
            return len(self.template_files) * 10  # 10 pairs for validation (faster)

    def _load_and_normalize_pcd(self, pcd_file):
        """Load .pcd and normalize using shape_bbox."""
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)
        points = normalize_shape_bbox(points)
        return points

    def __getitem__(self, index):
        # Get template file (cycle through templates)
        template_idx = index % len(self.template_files)
        category, pcd_file = self.template_files[template_idx]

        # Load and normalize template (use cache if available)
        if pcd_file in self.cached_templates:
            raw_points = self.cached_templates[pcd_file].copy()
        else:
            raw_points = self._load_and_normalize_pcd(pcd_file)

        # Sample points if point_limit specified
        if self.point_limit is not None and raw_points.shape[0] > self.point_limit:
            raw_points = random_sample_points(raw_points, self.point_limit)

        # Split reference and source (like ModelNet lines 141-142)
        ref_points = raw_points.copy()
        src_points = raw_points.copy()

        # Random transform to source (like ModelNet lines 153-155)
        transform = random_sample_transform(
            self.rotation_range,
            self.translation_range
        )
        inv_transform = inverse_transform(transform)
        src_points = apply_transform(src_points, inv_transform)

        # Save raw before augmentations
        raw_ref_points = ref_points.copy()
        raw_src_points = src_points.copy()

        # Crop with plane (like ModelNet lines 169-175)
        if self.augmentation_crop and self.point_keep_ratio is not None:
            ref_points = random_crop_point_cloud_with_plane(
                ref_points, keep_ratio=self.point_keep_ratio
            )
            src_points = random_crop_point_cloud_with_plane(
                src_points, keep_ratio=self.point_keep_ratio
            )

        # Random jitter (like ModelNet lines 203-205)
        if self.use_augmentation and self.aug_noise > 0:
            ref_points = random_jitter_points(
                ref_points, scale=0.01, noise_magnitude=self.aug_noise
            )
            src_points = random_jitter_points(
                src_points, scale=0.01, noise_magnitude=self.aug_noise
            )

        # Random shuffle (like ModelNet lines 208-209)
        ref_points = random_shuffle_points(ref_points)
        src_points = random_shuffle_points(src_points)

        # Compute overlap
        overlap = compute_overlap(ref_points, src_points, transform)

        # Build data dict
        data_dict = {
            'scene_name': f'{category}_{osp.basename(pcd_file)[:-4]}',
            'ref_frame': 0,
            'src_frame': 0,
            'overlap': overlap,
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'transform': transform.astype(np.float32),
        }

        return data_dict


def train_valid_data_loader(cfg, distributed):
    train_dataset = Real3DADPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        augmentation_crop=cfg.train.get('augmentation_crop', False),
        point_keep_ratio=cfg.train.get('point_keep_ratio', 0.7),
        rotation_range=cfg.train.rotation_range,
        translation_range=cfg.train.translation_range,
        matching_radius=cfg.train.matching_radius
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=True
    )

    valid_dataset = Real3DADPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        augmentation_crop=cfg.train.get('augmentation_crop', False),
        point_keep_ratio=cfg.train.get('point_keep_ratio', 0.7),
        rotation_range=cfg.train.rotation_range,
        translation_range=cfg.train.translation_range,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True
    )

    return train_loader, valid_loader, cfg.backbone.num_neighbors


def test_data_loader(cfg, benchmark):
    raise NotImplementedError("Test loader not needed for Real3DAD training")
