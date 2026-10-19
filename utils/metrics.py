"""
Metrics computation utilities for anomaly detection evaluation.

This module provides functions for:
- Image preprocessing (resize, blur) for MVTec3D-AD
- MVTec3D-AD sample processing for AU-PRO computation
- Metrics computation (I-AUROC, P-AUROC, AU-PRO)
"""
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score




def compute_metrics(image_preds, image_labels, pixel_preds=None, pixel_labels=None,
                   aupro_data=None, dataset_type='MVTec3DAD', category=''):
    """
    Compute and print evaluation metrics.

    Args:
        image_preds: Array of image-level predictions
        image_labels: Array of image-level labels
        pixel_preds: Array of pixel-level predictions (Real3D-AD only)
        pixel_labels: Array of pixel-level labels (Real3D-AD only)
        aupro_data: Tuple of (gts, predictions) for AU-PRO (MVTec3D-AD only)
        dataset_type: 'MVTec3DAD' or 'Real3DAD'
        category: Category name for display

    Returns:
        dict with metric values
    """
    image_rocauc = roc_auc_score(image_labels, image_preds)

    print(f"Category: {category}")
    print(f"Image ROCAUC: {image_rocauc:.3f}")

    metrics = {'image_rocauc': image_rocauc}

    if dataset_type == 'Real3DAD' or dataset_type == 'ShapeNetAD' and pixel_preds is not None:
        # Compute P-AUROC for Real3D-AD
        pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)
        print(f"Pixel ROCAUC: {pixel_rocauc:.3f}")
        metrics['pixel_rocauc'] = pixel_rocauc

    return metrics