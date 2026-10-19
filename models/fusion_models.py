"""
Fusion network models for FPFH + Geometry anomaly detection.
"""
import torch
import torch.nn as nn
from utils.geometry import compute_geometry_scores_learned_fusion
# Import optimized vectorized functions for faster inference
from utils.geometry_optimized import (
    compute_knn_once,
    compute_memory_bank_statistics_vectorized,
    compute_geometry_scores_learned_fusion_vectorized,
    compute_fpfh_scores_vectorized
)


class LearnedFusionBeforeMin(nn.Module):
    """
    LEARNED FUSION BEFORE MIN - Preserves surface coherence!

    Key innovation: Predict fusion weights from MEMORY BANK STATISTICS,
    then fuse BEFORE taking min to find best COHERENT surface match.

    Architecture:
    1. Compute local memory bank consistency statistics (variance of means/normals)
    2. MLP predicts weights [w_mean, w_normal] from these statistics
    3. Weights are applied BEFORE min: min(w_mean*mean_dists + w_normal*normal_diffs)

    Why this design is better:
    - Weights depend on LOCAL RELIABILITY of each metric in the memory bank
    - High variance in normals → trust mean distance more
    - High variance in means → trust normal alignment more
    - Spatial location captures regional reliability patterns

    Input features (9D):
    - centroid (3D): spatial location
    - mean_dist statistics (3D): min, mean, std among k nearest training means
    - normal_diff statistics (3D): min, mean, std among k nearest training normals
    """
    def __init__(self, input_dim=9, hidden_dim=64):
        super().__init__()

        # Weight prediction network (NO BatchNorm - causes distribution shift!)
        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout to compensate for no BN
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),  # Predict w_mean, w_normal
            nn.Softmax(dim=1)  # w_mean + w_normal = 1
        )

    def forward(
        self,
        patch_points_batch,   # (P, patch_size, 3) patch points
        patch_center_batch,   # (P, 3) patch centers
        reference_patch_centers,
        reference_means,
        reference_normals,
        k_centers=10
    ):
        """
        Forward pass with learned fusion BEFORE min.

        Args:
            patch_points_batch: (P, patch_size, 3) test patches
            patch_center_batch: (P, 3) test centers
            reference_patch_centers: (R, 3) reference centers
            reference_means: (R, N, 3) reference means
            reference_normals: (R, N, 3) reference normals
            k_centers: Number of nearest centers to use

        Returns:
            scores: (P,) anomaly scores (min of weighted combination)
        """
        from utils.geometry import compute_memory_bank_statistics

        # Compute memory bank statistics for fusion weight prediction
        # Features: centroid (3D) + mean_stats (3D) + normal_stats (3D) = 9D
        reliability_features = compute_memory_bank_statistics(
            patch_points_batch,
            patch_center_batch,
            reference_patch_centers,
            reference_means,
            reference_normals,
            k_centers=k_centers
        )  # (P, 9)

        # Predict fusion weights from memory bank reliability
        fusion_weights = self.weight_predictor(reliability_features.float())  # (P, 2)

        # Compute scores with learned fusion BEFORE min
        scores = compute_geometry_scores_learned_fusion(
            patch_points_batch,
            patch_center_batch,
            reference_patch_centers,
            reference_means,
            reference_normals,
            fusion_weights,
            k_centers=k_centers
        )

        return scores


class SigmoidFusion(nn.Module):
    """
    Hierarchical fusion with SIGMOID gates (instead of softmax).

    Architecture (same as HierarchicalFusion but with sigmoid):
    - Level 1 (Early): Geometry fusion (mean_dist + normal_diff) → geometry_score
      Uses LearnedFusionBeforeMin (same as HierarchicalFusion)
    - Level 2 (Late): Score fusion (geometry_score + fpfh_score) → final_score
      Uses SIGMOID gates (different from HierarchicalFusion's softmax)
    """
    def __init__(self, patch_feature_dim=7, hidden_dim=16):
        """
        Args:
            patch_feature_dim: Dimension of patch features 
            hidden_dim: Hidden layer size 
        """
        super().__init__()
        hidden_dim_early = 16
        # Level 1: Early geometry fusion (mean + normal) 
        self.geometry_fusion = LearnedFusionBeforeMin(
            input_dim=9,  
            hidden_dim=hidden_dim_early
        )

        # Level 2: Late fusion with SIGMOID gates - DIFFERENT from HierarchicalFusion
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
        Forward pass with hierarchical fusion using SIGMOID gates.
        """
        nearest_indices, center_dists = compute_knn_once(
            patch_center_batch, reference_patch_centers, k_centers
        )  # (P, k), (P, R)

        min_center_dist, nearest_idx = center_dists.min(dim=1)  # (P,), (P,)
        closest_ref_centers = reference_patch_centers[nearest_idx]  # (P, 3)

        patch_features = torch.cat([
            patch_center_batch,      # (P, 3)
            closest_ref_centers,     # (P, 3)
            min_center_dist.unsqueeze(1)  # (P, 1)
        ], dim=1)  # (P, 7)

        # ========== Compute FPFH score ==========
        score_fpfh, _ = compute_fpfh_scores_vectorized(
            patch_fpfh_batch,
            reference_patch_fpfh,
            nearest_indices  # Reuse KNN!
        )  # (P,)

        # ========== Level 1: Geometry fusion==========
        # Compute memory bank statistics 
        reliability_features = compute_memory_bank_statistics_vectorized(
            patch_points_batch,
            patch_center_batch,
            reference_means,
            reference_normals,
            nearest_indices
        )  # (P, 9)

        # Predict fusion weights using original MLP
        fusion_weights = self.geometry_fusion.weight_predictor(reliability_features.float())  # (P, 2)

        # Compute geometry scores with learned fusion (vectorized, no loops)
        geometry_scores = compute_geometry_scores_learned_fusion_vectorized(
            patch_points_batch,
            reference_means,
            reference_normals,
            fusion_weights,
            nearest_indices  
        )  

        # final = exp(w1*log(geo) + w2*log(fpfh))
        eps = 1e-6 
        # Log transform both scores
        log_geo = torch.log(geometry_scores + eps)  # (P,)
        log_fpfh = torch.log(score_fpfh + eps)      # (P,)
        # Stack log scores
        log_scores = torch.stack([log_geo.float(), log_fpfh.float()], dim=1)  # (P, 2)

        # Level 2: Late fusion with sigmoid gates
        # Concatenate log scores + patch features
        late_fusion_input = torch.cat([
            log_scores.float(),
            patch_features.float()
        ], dim=1)  # (P, 2+7=9)

        # Predict independent weights via sigmoid (acts as exponents in geometric mean)
        weights = self.late_fusion(late_fusion_input)  # (P, 2), each in [0, 1]

        # Weighted combination in log space: w1*log(geo) + w2*log(fpfh)
        log_final = (weights * log_scores).sum(dim=1)  # (P,)

        # Exponentiate to get final score: exp(w1*log(geo) + w2*log(fpfh)) = geo^w1 * fpfh^w2
        final_scores = torch.exp(log_final)  # (P,)

        return final_scores
