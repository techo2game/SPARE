import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm
import open3d as o3d

from utils.util import normalize, random_rorate, random_patch, random_translate

all_shapenetad_cates = ['ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 'bowl0', 'bowl1', 'bowl2', 'bowl3', 'bowl4', 'bowl5', 'bucket0', 'bucket1', 'cap0', 'cap3', 'cap4', 'cap5', 'cup0', 'cup1', 'eraser0', 'headset0', 'headset1', 'helmet0', 'helmet1', 'helmet2', 'helmet3', 'jar0', 'microphone0', 'shelf0', 'tap0', 'tap1', 'vase0', 'vase1', 'vase2', 'vase3', 'vase4', 'vase5', 'vase7', 'vase8', 'vase9']

all_real3dad_cates = ['airplane', 'car', 'candybar', 'chicken', 'diamond', 'duck', 'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees']

all_mvtec3dad_cates = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']


class ShapeNetAD(Dataset):
    
    def __init__(self, path, cates, split, scale_mode=None, num_points=2048, num_aug=4, transforms=list()):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            self.cates = all_shapenetad_cates
        else:
            self.cates = cates
        self.split = split
        self.scale_mode = scale_mode
        self.num_points = num_points
        self.num_aug = num_aug
        self.transforms = transforms
        self.gt_dir_name = 'GT'  # ShapeNetAD uses uppercase 'GT'

        self.pointclouds = []
        self.stats = None

        # self.get_statistics()
        self.load()

    def get_statistics(self):

        stats_dir = os.path.join(self.path, '../shapenet-ad' + '_stats/')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cates) == len(all_shapenetad_cates):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cates) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        pointclouds = []
        for cate in self.cates:
            for split in ('train', 'test'):
                local_path = os.path.join(self.path, cate, split)
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, np.float32)
                    choice = np.random.permutation(len(pointcloud))[:self.num_points]
                    pointcloud = torch.from_numpy(pointcloud[choice])
                    pointclouds.append(pointcloud)

        all_points = torch.stack(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def scale(self, pc):
        if self.scale_mode == 'global_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_34':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        pc = (pc - shift) / scale
        
        return pc, shift, scale

    def append(self, pc, cate, pc_id, mask, label):
        pc, shift, scale = self.scale(pc)
        self.pointclouds.append({
            'pointcloud': pc,
            'cate': cate,
            'id': pc_id,
            'shift': shift,
            'scale': scale,
            'mask': mask,
            'label': label,
        })
    
    def load(self):

        for cate in self.cates:
            if self.split == 'train':
                local_path = os.path.join(self.path, cate, 'train')
                # Skip category if train directory doesn't exist
                if not os.path.exists(local_path):
                    print(f"Warning: Skipping category '{cate}' - train directory not found: {local_path}")
                    continue
                tpls = []
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, dtype=np.float32)
                    tpls.append(pointcloud)
                for pc_id in tqdm(range(self.num_aug), 'Augment'):
                    if self.num_aug == len(tpls):
                        pointcloud = tpls[pc_id]
                    else:
                        pointcloud = random.choice(tpls)
                    pointcloud = random_rorate(pointcloud)
                    choice = np.random.permutation(len(pointcloud))[:self.num_points]
                    pc = torch.from_numpy(pointcloud[choice])
                    mask = torch.zeros(self.num_points)
                    label = 0
                    self.append(pc, cate, pc_id, mask, label)

            elif self.split == 'test':
                local_path = os.path.join(self.path, cate, 'test')
                # Skip category if test directory doesn't exist
                if not os.path.exists(local_path):
                    print(f"Warning: Skipping category '{cate}' - test directory not found: {local_path}")
                    continue
                for pc_id, f in enumerate(os.listdir(local_path)):
                    if 'positive' in f:
                        local_file = os.path.join(local_path, f)
                        pcd = o3d.io.read_point_cloud(local_file)
                        pointcloud = np.array(pcd.points, dtype=np.float32)
                        # Sample only if we have more points than requested
                        if len(pointcloud) > self.num_points:
                            choice = np.random.permutation(len(pointcloud))[:self.num_points]
                            pointcloud = pointcloud[choice]
                        pc = torch.from_numpy(pointcloud)
                        mask = torch.zeros(pc.shape[0])
                        label = 0
                    else:
                        local_file = os.path.join(local_path.replace('test', self.gt_dir_name), f.replace('pcd', 'txt'))
                        pointcloud_mask = np.loadtxt(local_file, dtype=np.float32, delimiter=",")
                        # Sample only if we have more points than requested
                        if len(pointcloud_mask) > self.num_points:
                            choice = np.random.permutation(len(pointcloud_mask))[:self.num_points]
                            pointcloud_mask = pointcloud_mask[choice]
                        pc = torch.from_numpy(pointcloud_mask[:, :3])
                        mask = torch.from_numpy(pointcloud_mask[:, 3])
                        label = 1
                    self.append(pc, cate, pc_id, mask, label)

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        for transform in self.transforms:
            data = transform(data)
        return data

class Real3DAD(Dataset):

    def __init__(self, path, cates, split, scale_mode=None, num_points=2048, num_aug=4, transforms=list()):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            self.cates = all_real3dad_cates
        else:
            self.cates = cates
        self.split = split
        self.scale_mode = scale_mode
        self.num_points = num_points
        self.num_aug = num_aug
        self.transforms = transforms
        self.gt_dir_name = 'gt'  # Real3DAD uses lowercase 'gt'

        self.pointclouds = []
        self.stats = None

        # self.get_statistics()
        self.load()

    def get_statistics(self):

        stats_dir = os.path.join(self.path, '../real3d-ad' + '_stats/')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cates) == len(all_real3dad_cates):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cates) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        pointclouds = []
        for cate in self.cates:
            for split in ('train', 'test'):
                local_path = os.path.join(self.path, cate, split)
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, np.float32)
                    choice = np.random.permutation(len(pointcloud))[:self.num_points]
                    pointcloud = torch.from_numpy(pointcloud[choice])
                    pointclouds.append(pointcloud)

        all_points = torch.stack(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def scale(self, pc):
        if self.scale_mode == 'global_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_34':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        pc = (pc - shift) / scale

        return pc, shift, scale

    def append(self, pc, cate, pc_id, mask, label):
        pc, shift, scale = self.scale(pc)
        self.pointclouds.append({
            'pointcloud': pc,
            'cate': cate,
            'id': pc_id,
            'shift': shift,
            'scale': scale,
            'mask': mask,
            'label': label,
        })

    def load(self):

        for cate in self.cates:
            if self.split == 'train':
                local_path = os.path.join(self.path, cate, 'train')
                # Skip category if train directory doesn't exist
                if not os.path.exists(local_path):
                    print(f"Warning: Skipping category '{cate}' - train directory not found: {local_path}")
                    continue
                tpls = []
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, dtype=np.float32)
                    tpls.append(pointcloud)
                for pc_id in tqdm(range(self.num_aug), 'Augment'):
                    if self.num_aug == len(tpls):
                        pointcloud = tpls[pc_id]
                    else:
                        pointcloud = random.choice(tpls)
                    pointcloud = random_rorate(pointcloud)
                    choice = np.random.permutation(len(pointcloud))[:self.num_points]
                    pc = torch.from_numpy(pointcloud[choice])
                    mask = torch.zeros(self.num_points)
                    label = 0
                    self.append(pc, cate, pc_id, mask, label)

            elif self.split == 'test':
                local_path = os.path.join(self.path, cate, 'test')

                for pc_id, f in enumerate(os.listdir(local_path)):
                    # Check if it's a normal (good/positive) sample
                    if 'good' in f:
                        local_file = os.path.join(local_path, f)
                        pcd = o3d.io.read_point_cloud(local_file)
                        pointcloud = np.array(pcd.points, dtype=np.float32)
                        # Sample only if we have more points than requested
                        if len(pointcloud) > self.num_points:
                            choice = np.random.permutation(len(pointcloud))[:self.num_points]
                            pointcloud = pointcloud[choice]
                        pc = torch.from_numpy(pointcloud)
                        mask = torch.zeros(pc.shape[0])
                        label = 0
                    else:
                        # Load GT file: (x,y,z,gt)
                        gt_file = os.path.join(local_path.replace('test', self.gt_dir_name), f.replace('.pcd', '.txt'))
                        pointcloud_mask = np.loadtxt(gt_file, dtype=np.float32)
                        # Sample only if we have more points than requested
                        if len(pointcloud_mask) > self.num_points:
                            choice = np.random.permutation(len(pointcloud_mask))[:self.num_points]
                            pointcloud_mask = pointcloud_mask[choice]
                        pc = torch.from_numpy(pointcloud_mask[:, :3])
                        mask = torch.from_numpy(pointcloud_mask[:, 3])
                        label = 1
                    self.append(pc, cate, pc_id, mask, label)

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        for transform in self.transforms:
            data = transform(data)
        return data
