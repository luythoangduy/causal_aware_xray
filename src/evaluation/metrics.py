from __future__ import annotations
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_auc_metrics(y_true: Tensor, y_pred: Tensor, pathologies: List[str]) -> Dict[str, float]:
    """Compute AUC metrics for multi-label classification."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    results = {}
    
    # Per-class AUC
    for i, pathology in enumerate(pathologies):
        if y_true_np[:, i].sum() > 0:  # Only if positive samples exist
            try:
                auc = roc_auc_score(y_true_np[:, i], y_pred_np[:, i])
                results[f'auc_{pathology.lower()}'] = auc
            except ValueError:
                results[f'auc_{pathology.lower()}'] = 0.5
    
    # Macro AUC
    class_aucs = [v for k, v in results.items() if k.startswith('auc_')]
    if class_aucs:
        results['auc_macro'] = np.mean(class_aucs)
    
    # Micro AUC
    try:
        results['auc_micro'] = roc_auc_score(y_true_np.ravel(), y_pred_np.ravel())
    except ValueError:
        results['auc_micro'] = 0.5
        
    return results


def compute_ap_metrics(y_true: Tensor, y_pred: Tensor, pathologies: List[str]) -> Dict[str, float]:
    """Compute Average Precision metrics."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    results = {}
    
    # Per-class AP
    for i, pathology in enumerate(pathologies):
        if y_true_np[:, i].sum() > 0:
            try:
                ap = average_precision_score(y_true_np[:, i], y_pred_np[:, i])
                results[f'ap_{pathology.lower()}'] = ap
            except ValueError:
                results[f'ap_{pathology.lower()}'] = y_true_np[:, i].mean()
    
    # Macro AP
    class_aps = [v for k, v in results.items() if k.startswith('ap_')]
    if class_aps:
        results['ap_macro'] = np.mean(class_aps)
    
    # Micro AP
    try:
        results['ap_micro'] = average_precision_score(y_true_np.ravel(), y_pred_np.ravel())
    except ValueError:
        results['ap_micro'] = y_true_np.mean()
        
    return results


def compute_classification_metrics(y_true: Tensor, y_pred: Tensor, pathologies: List[str], 
                                 threshold: float = 0.5) -> Dict[str, float]:
    """Compute precision, recall, F1 metrics at given threshold."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = (y_pred.detach().cpu().numpy() > threshold).astype(int)
    
    results = {}
    
    # Per-class metrics
    for i, pathology in enumerate(pathologies):
        if y_true_np[:, i].sum() > 0:
            try:
                precision = precision_score(y_true_np[:, i], y_pred_np[:, i], zero_division=0)
                recall = recall_score(y_true_np[:, i], y_pred_np[:, i], zero_division=0)
                f1 = f1_score(y_true_np[:, i], y_pred_np[:, i], zero_division=0)
                
                results[f'precision_{pathology.lower()}'] = precision
                results[f'recall_{pathology.lower()}'] = recall
                results[f'f1_{pathology.lower()}'] = f1
            except ValueError:
                results[f'precision_{pathology.lower()}'] = 0.0
                results[f'recall_{pathology.lower()}'] = 0.0
                results[f'f1_{pathology.lower()}'] = 0.0
    
    # Macro averages
    precisions = [v for k, v in results.items() if k.startswith('precision_')]
    recalls = [v for k, v in results.items() if k.startswith('recall_')]
    f1s = [v for k, v in results.items() if k.startswith('f1_')]
    
    if precisions:
        results['precision_macro'] = np.mean(precisions)
        results['recall_macro'] = np.mean(recalls) 
        results['f1_macro'] = np.mean(f1s)
    
    # Micro averages
    try:
        results['precision_micro'] = precision_score(y_true_np.ravel(), y_pred_np.ravel(), zero_division=0)
        results['recall_micro'] = recall_score(y_true_np.ravel(), y_pred_np.ravel(), zero_division=0)
        results['f1_micro'] = f1_score(y_true_np.ravel(), y_pred_np.ravel(), zero_division=0)
    except ValueError:
        results['precision_micro'] = 0.0
        results['recall_micro'] = 0.0
        results['f1_micro'] = 0.0
    
    return results


def compute_constraint_metrics(y_pred: Tensor, constraint_layer) -> Dict[str, float]:
    """Compute constraint violation metrics."""
    if constraint_layer is None:
        return {}
    
    violations = constraint_layer.constraint_violations(y_pred)
    
    # Normalize by batch size
    batch_size = y_pred.shape[0]
    normalized_violations = {}
    
    for key, value in violations.items():
        if 'total' in key:
            # Convert to per-sample average
            normalized_key = key.replace('total', 'avg')
            normalized_violations[normalized_key] = value / batch_size if batch_size > 0 else 0.0
        else:
            normalized_violations[key] = value
    
    return normalized_violations


def compute_all_metrics(y_true: Tensor, y_pred: Tensor, pathologies: List[str], 
                       constraint_layer=None, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {}
    
    # AUC metrics
    metrics.update(compute_auc_metrics(y_true, y_pred, pathologies))
    
    # AP metrics  
    metrics.update(compute_ap_metrics(y_true, y_pred, pathologies))
    
    # Classification metrics
    metrics.update(compute_classification_metrics(y_true, y_pred, pathologies, threshold))
    
    # Constraint metrics
    metrics.update(compute_constraint_metrics(y_pred, constraint_layer))
    
    return metrics


# Legacy function for backward compatibility
def compute_basic_metrics(y_true: np.ndarray, y_prob: np.ndarray, pathologies: List[str]) -> Dict[str, float]:
    """Legacy function - use compute_all_metrics instead."""
    y_true_tensor = torch.from_numpy(y_true)
    y_prob_tensor = torch.from_numpy(y_prob)
    
    results = {}
    # Per-class AUC & AP
    aucs = []
    aps = []
    for i, _ in enumerate(pathologies):
        try:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            aucs.append(float('nan'))
        try:
            aps.append(average_precision_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            aps.append(float('nan'))
    results['auc_macro'] = np.nanmean(aucs)
    results['ap_macro'] = np.nanmean(aps)

    # Micro AUC (flatten)
    try:
        results['auc_micro'] = roc_auc_score(y_true.ravel(), y_prob.ravel())
    except ValueError:
        results['auc_micro'] = float('nan')

    return results
