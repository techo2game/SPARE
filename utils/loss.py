"""
Better loss functions for fusion network training.

The original ranking loss has mode collapse: the model learns x1=x2≈0
to minimize loss without actually separating classes.

These losses force the model to maintain separation between classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_contrastive_loss(final_score, labels, margin=1.0, num_pairs=2000, hard_mining=False):
    """

    Args:
        final_score: (N,) predicted anomaly scores
        labels: (N,) binary labels (0=normal, 1=anomaly)
        margin: minimum separation between anomaly and normal scores (default: 1.0)
        num_pairs: number of (anomaly, normal) pairs to sample per batch (default: 2000)
        hard_mining: if True, focus on hardest examples (lowest anomalies, highest normals)

    Returns:
        loss: scalar tensor
    """

    anom_idx = (labels == 1).nonzero(as_tuple=True)[0]
    norm_idx = (labels == 0).nonzero(as_tuple=True)[0]


    if len(anom_idx) == 0 or len(norm_idx) == 0:

        return F.binary_cross_entropy_with_logits(final_score, labels)

    # Determine number of pairs to sample
    max_possible_pairs = len(anom_idx) * len(norm_idx)
    num_pairs_actual = min(num_pairs, max_possible_pairs)

    if hard_mining:

        anom_scores = final_score[anom_idx]
        norm_scores = final_score[norm_idx]


        num_hard = max(1, len(anom_idx) // 2)
        num_easy = max(1, len(norm_idx) // 2)


        hard_norm_idx = torch.topk(norm_scores, min(num_easy, len(norm_idx)), largest=True)[1]
        hard_norm_global = norm_idx[hard_norm_idx]


        easy_anom_idx = torch.topk(anom_scores, min(num_hard, len(anom_idx)), largest=False)[1]
        easy_anom_global = anom_idx[easy_anom_idx]


        num_pairs_actual = min(num_pairs, len(easy_anom_global) * len(hard_norm_global))
        anom_samples = easy_anom_global[torch.randint(0, len(easy_anom_global), (num_pairs_actual,), device=final_score.device)]
        norm_samples = hard_norm_global[torch.randint(0, len(hard_norm_global), (num_pairs_actual,), device=final_score.device)]
    else:

        anom_samples = anom_idx[torch.randint(0, len(anom_idx), (num_pairs_actual,), device=final_score.device)]
        norm_samples = norm_idx[torch.randint(0, len(norm_idx), (num_pairs_actual,), device=final_score.device)]

    # Get scores for sampled pairs
    anom_scores_sampled = final_score[anom_samples]
    norm_scores_sampled = final_score[norm_samples]


    pairwise_loss = F.relu(margin - (anom_scores_sampled - norm_scores_sampled))

    return pairwise_loss.mean()


def ranking_loss_with_regularization(final_score, labels, reg_weight=0.01):
    """
    Improved ranking loss with regularization to prevent weight collapse.

    Args:
        final_score: (N,) predicted anomaly scores
        labels: (N,) binary labels (0=normal, 1=anomaly)
        reg_weight: weight for regularization term (default: 0.01)

    Adds penalty for very small scores to prevent x1=x2→0 collapse.
    """
    # Original ranking loss
    ranking = ((1 - 2 * labels) * final_score).mean()

    # Regularization: penalize scores near zero (prevents collapse)
    # We want to encourage non-zero scores
    score_magnitude = final_score.abs().mean()
    reg_loss = -torch.log(score_magnitude + 1e-8)  # Negative log encourages larger magnitude

    return ranking + reg_weight * reg_loss


def get_loss_function(loss_type='bce'):
    """
    Factory function to get loss function by name.

    Args:
        loss_type: 'bce', 'focal', 'contrastive', 'ranking_reg', 'auc', or 'patch_contrastive'

    Returns:
        loss function
    """
    loss_functions = {
        'patch_contrastive': patch_contrastive_loss
    }

    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_functions.keys())}")

    return loss_functions[loss_type]


