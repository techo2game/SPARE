"""
Seed-based synthetic anomaly generation.

New architecture (mesh-aware sculpting):
- Seeds define WHERE to sculpt (center/spine of deformation)
- Mesh-aware deformation maintains surface continuity
- Natural falloff from seed (no separate propagation step)
- Distance-based ground truth
"""

import numpy as np
from utils.seed_generators import get_seed_generator
from utils.deformation_methods import apply_seed_based_deformation


class AnomalyGenerator:
    """
    Seed-based anomaly generator with mesh-aware sculpting.

    Workflow:
    1. Generate seeds (line or point seeds) - defines WHERE to sculpt
    2. Apply mesh-aware sculpting deformation with natural falloff
    3. Compute distance-based ground truth

    Args:
        config: Configuration dict with 'seed' and 'deformation' sections
            seed.type: 'line_fps', 'point_fps', 'point_random'
            seed.num_seeds: Number of seeds
            seed.line_length: Line length (for line seeds)
            deformation.influence_radius: Deformation radius
            deformation.amplitude: Deformation strength
            deformation.falloff_type: 'gaussian', 'linear', 'cosine'
            deformation.smoothing_iterations: Mesh smoothing iterations
            deformation.distance_threshold: GT distance threshold
    """

    def __init__(self, config):
        """
        Initialize seed-based anomaly generator.

        Args:
            config: Configuration dict with seed and deformation sections
        """
        # Extract configuration sections
        if 'seed' not in config:
            raise ValueError("Config must have 'seed' section")
        if 'deformation' not in config:
            raise ValueError("Config must have 'deformation' section")

        self.seed_cfg = config['seed'].copy()
        self.deform_cfg = config['deformation'].copy()

        # Initialize seed generator
        seed_type = self.seed_cfg.get('type', 'point_fps')
        self.seed_generator = get_seed_generator(seed_type)

        # Optional: probability of applying anomaly
        self.p = config.get('p', 1.0)

    def compute(self, pointcloud):
        """
        Generate synthetic anomaly using mesh-aware sculpting.

        Args:
            pointcloud: (N, 3) numpy array

        Returns:
            anomalous_pc: (N, 3) numpy array (deformed)
            binary_mask: (N,) binary mask (0=normal, 1=anomaly)
        """
        # Probabilistic application
        if np.random.rand() > self.p:
            return pointcloud.copy(), np.zeros(len(pointcloud), dtype=np.float32)

        # Step 1: Generate seeds (defines WHERE to sculpt)
        seed_params = {k: v for k, v in self.seed_cfg.items() if k != 'type'}
        seed_data = self.seed_generator(pointcloud, **seed_params)

        # Step 2: Apply mesh-aware sculpting deformation
        # Deformation has natural falloff and maintains surface continuity
        anomalous_pc, binary_mask = apply_seed_based_deformation(
            pointcloud,
            seed_data,
            **self.deform_cfg
        )

        return anomalous_pc, binary_mask

    def __repr__(self):
        seed_type = self.seed_cfg.get('type', 'unknown')
        falloff = self.deform_cfg.get('falloff_type', 'gaussian')
        return f"AnomalyGenerator(seed={seed_type}, falloff={falloff})"
