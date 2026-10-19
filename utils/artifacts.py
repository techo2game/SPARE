"""Artifact path helpers shared by scripts.

All generated artifacts live under the repository root (see README, "Pipeline").
"""
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def dataset_dir_name(dataset):
    mapping = {
        'Real3DAD': 'real3d-ad',
        'ShapeNetAD': 'shapenet-ad',
        'MVTec3DAD': 'mvtec3d-ad',
    }
    return mapping.get(dataset, str(dataset).lower())


def first_existing_path(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def registration_train_dir(dataset, category):
    dataset_dir = dataset_dir_name(dataset)
    return first_existing_path(os.path.join(PROJECT_ROOT, 'reg_output', dataset_dir, category, 'train'))


def reference_sample_path(dataset, category):
    reg_dir = registration_train_dir(dataset, category)
    if reg_dir is None:
        return None
    return os.path.join(reg_dir, 'sample_000.pkl')


def spatial_chamfer_bank_path(category):
    return first_existing_path(
        os.path.join(PROJECT_ROOT, 'spatial_memory_banks', f'{category}_spatial_chamfer_memory_bank.pt')
    )
