"""PARENet registration wrapper for test_variability framework.

Wraps PARENet (deep learning registration) to match the standard interface.
Uses singleton pattern to load model once and reuse across calls.
"""
import os
import sys
import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Make the PARENet submodule and the SPARE Real3DAD experiment importable
sys.path.insert(0, os.path.join(REPO_ROOT, 'parenet_real3dad'))
import _paths  # noqa: E402,F401

# Fine-tuned PARENet weights (see README: "PARENet registration model")
DEFAULT_CHECKPOINT = os.environ.get(
    'SPARE_PARENET_CHECKPOINT',
    os.path.join(REPO_ROOT, 'pretrain', 'parenet.tar'),
)

from config import make_cfg
from model import create_model
from dataset import normalize_shape_bbox
from pareconv.utils.data import registration_collate_fn_stack_mode, precompute_neibors
from pareconv.utils.torch import release_cuda, to_cuda


# Global singleton for model
_PARENET_MODEL = None
_PARENET_CFG = None


def _load_parenet_model(checkpoint_path=None):
    """Load PARENet model (singleton pattern)."""
    global _PARENET_MODEL, _PARENET_CFG

    if _PARENET_MODEL is not None:
        return _PARENET_MODEL, _PARENET_CFG

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
        print(f'Using default PARENet checkpoint: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'PARENet checkpoint not found: {checkpoint_path}. Train one with '
            '`python parenet_real3dad/trainval.py` and copy it to pretrain/parenet.tar '
            '(or set SPARE_PARENET_CHECKPOINT).'
        )

    # Load config
    cfg = make_cfg()

    # Create model
    model = create_model(cfg).cuda()
    model.eval()

    # Load checkpoint
    print(f'Loading PARENet checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint.get('epoch', 0)
    print(f'Loaded PARENet model from epoch {epoch}')

    _PARENET_MODEL = model
    _PARENET_CFG = cfg

    return model, cfg


def _prepare_parenet_input(test, train, cfg):
    """Convert raw tensors to PARENet input format.

    Args:
        test: (N, 3) tensor on GPU
        train: (M, 3) tensor on GPU
        cfg: PARENet config

    Returns:
        data_dict: Dict ready for PARENet forward pass
    """
    # Convert to numpy for compatibility with collate function
    test_np = test.cpu().numpy()
    train_np = train.cpu().numpy()

    # Subsample to point_limit if needed
    point_limit = cfg.test.point_limit  # 30000
    if test_np.shape[0] > point_limit:
        indices = np.random.permutation(test_np.shape[0])[:point_limit]
        test_np = test_np[indices]
    if train_np.shape[0] > point_limit:
        indices = np.random.permutation(train_np.shape[0])[:point_limit]
        train_np = train_np[indices]

    # Create data dict (same format as dataset.py)
    data_dict = {
        'ref_points': train_np.astype(np.float32),
        'src_points': test_np.astype(np.float32),
        'ref_feats': np.ones((train_np.shape[0], 1), dtype=np.float32),
        'src_feats': np.ones((test_np.shape[0], 1), dtype=np.float32),
        'transform': np.eye(4, dtype=np.float32),
        'scene_name': 'test_pair',
        'ref_frame': 0,
        'src_frame': 0,
        'overlap': 0.0,
    }

    # Apply collate function (multi-scale preprocessing)
    data_list = [data_dict]
    batch = registration_collate_fn_stack_mode(
        data_list,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        precompute_data=True
    )

    # Move to GPU
    batch = to_cuda(batch)

    # Compute neighbors (kNN search)
    neighbor_data = precompute_neibors(
        batch['points'],
        batch['lengths'],
        cfg.backbone.num_stages,
        cfg.backbone.num_neighbors
    )
    batch.update(neighbor_data)

    return batch


def parenet_torch(test, train, max_iter=200, num_points=2000, lr=0.1,
                  use_scheduler=True, num_restarts=1,
                  checkpoint_path=None, **kwargs):
    """PARENet registration for test_variability framework.

    Args:
        test: Test point cloud (N, 3) PyTorch tensor on GPU
        train: Reference point cloud (M, 3) PyTorch tensor on GPU
        max_iter: Ignored (PARENet is single-pass, no optimization)
        num_points: Ignored (PARENet uses cfg.test.point_limit = 30000)
        lr: Ignored
        use_scheduler: Ignored
        num_restarts: Ignored (PARENet generates hypotheses internally)
        checkpoint_path: Path to .pth.tar checkpoint (optional)
        **kwargs: Ignored

    Returns:
        T: 4x4 transformation matrix as PyTorch tensor on device
    """
    device = test.device

    # Load model (singleton)
    model, cfg = _load_parenet_model(checkpoint_path)

    # Prepare input
    data_dict = _prepare_parenet_input(test, train, cfg)

    # Forward pass
    with torch.no_grad():
        output_dict = model(data_dict)

    # Extract transformation
    estimated_transform = release_cuda(output_dict['estimated_transform'])  # (4, 4) numpy

    # Convert to PyTorch tensor on device
    T = torch.from_numpy(estimated_transform.astype(np.float32)).to(device)

    return T
