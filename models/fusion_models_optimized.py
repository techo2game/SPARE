"""
Optimized fusion models with vectorized operations.

Key optimization: Compute KNN ONCE, reuse across all feature computations.
Expected speedup: 650ms → 50-80ms (8-13× faster)
"""
import torch
import torch.nn as nn
from models.fusion_models import LearnedFusionBeforeMin
from utils.geometry_optimized import (
    compute_knn_once,
    compute_memory_bank_statistics_vectorized,
    compute_geometry_scores_learned_fusion_vectorized,
    compute_fpfh_scores_vectorized
)


class SigmoidFusionOptimized(nn.Module):
    """
    Optimized SigmoidFusion with vectorized operations.
    - Level 1: Geometry fusion (mean + normal) → geometry_score
    - Level 2: Score fusion (geometry + FPFH) → final_score with sigmoid gates
    """
    def __init__(self, patch_feature_dim=7, hidden_dim=16):
        super().__init__()
        hidden_dim_early = 16

        # Level 1: Geometry fusion
        self.geometry_fusion_mlp = LearnedFusionBeforeMin(
            input_dim=9,
            hidden_dim=hidden_dim_early
        ).weight_predictor 

        # Level 2: Late fusion (same as original)
        fusion_input_dim = 2 + patch_feature_dim

        self.late_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )

    def forward(
        self,
        patch_points_batch,       # (P, patch_size, 3)
        patch_fpfh_batch,         # (P, 33)
        patch_center_batch,       # (P, 3)
        reference_patch_centers,  # (R, 3)
        reference_means,          # (R, N, 3)
        reference_normals,        # (R, N, 3)
        reference_patch_fpfh,     # (R, N, 33)
        k_centers=10
    ):
        """
        Optimized forward pass 
        """

        nearest_indices, _ = compute_knn_once(
            patch_center_batch, reference_patch_centers, k_centers
        )  # (P, k)


        center_dists = torch.cdist(patch_center_batch.float(), reference_patch_centers.float())
        min_center_dist, nearest_idx = center_dists.min(dim=1)  # (P,), (P,)
        closest_ref_centers = reference_patch_centers[nearest_idx]  # (P, 3)

        patch_features = torch.cat([
            patch_center_batch,      # (P, 3)
            closest_ref_centers,     # (P, 3)
            min_center_dist.unsqueeze(1)  # (P, 1)
        ], dim=1)  # (P, 7)


        score_fpfh, _ = compute_fpfh_scores_vectorized(
            patch_fpfh_batch,
            reference_patch_fpfh,
            nearest_indices  
        )  
        # ========== Level 1: Geometry fusion =========

        reliability_features = compute_memory_bank_statistics_vectorized(
            patch_points_batch,
            patch_center_batch,
            reference_means,
            reference_normals,
            nearest_indices  # Reuse KNN!
        )  # (P, 9)

        # Predict fusion weights using original MLP
        fusion_weights = self.geometry_fusion_mlp(reliability_features.float())  # (P, 2)

        # Compute geometry scores with learned fusion
        geometry_scores = compute_geometry_scores_learned_fusion_vectorized(
            patch_points_batch,
            reference_means,
            reference_normals,
            fusion_weights,
            nearest_indices 
        )  # (P,)

        # ========== Level 2: Late fusion==========
        # Multiplicative fusion in log space
        eps = 1e-6
        log_geo = torch.log(geometry_scores + eps)
        log_fpfh = torch.log(score_fpfh + eps)

        log_scores = torch.stack([log_geo.float(), log_fpfh.float()], dim=1)  # (P, 2)

        late_fusion_input = torch.cat([
            log_scores.float(),
            patch_features.float()
        ], dim=1)  # (P, 9)

        weights = self.late_fusion(late_fusion_input)  # (P, 2)
        log_final = (weights * log_scores).sum(dim=1)  # (P,)
        final_scores = torch.exp(log_final)  # (P,)

        return final_scores