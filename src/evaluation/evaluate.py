from __future__ import annotations
import argparse
import os
import torch
import yaml
import wandb
import json
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from src.data.chestxray14 import ChestXRay14Dataset, PATHOLOGIES
from src.models.constraint_projection import ConstraintProjection
from src.models.classifier import ConstrainedChestXRayClassifier
from src.evaluation.metrics import compute_all_metrics
from src.training.train import MEDICAL_CONSTRAINTS, build_constraint_layer


def load_model(checkpoint_path, device, pathologies):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint if available
    config = checkpoint.get('config', {})
    
    # Build model
    constraint_layer = build_constraint_layer(pathologies, config)
    model = ConstrainedChestXRayClassifier(
        backbone=config.get('backbone', 'densenet121'),
        pretrained=True,
        num_classes=len(pathologies),
        constraint_projection=constraint_layer,
        projection_mode=config.get('projection_mode', 'integrated')
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_model(model, loader, device, pathologies):
    """Comprehensive model evaluation."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_raw_logits = []
    all_constraint_stats = []
    
    print("Running evaluation...")
    for batch in tqdm(loader):
        imgs = batch['img'].to(device)
        labs = batch['lab'].to(device)
        
        # Get raw logits and constrained predictions
        raw_logits = model(imgs, apply_constraints=False)
        constrained_probs = model(imgs)  # Uses integrated mode
        
        # Get constraint violations
        if model.constraint_projection:
            constraint_stats = model.constraint_projection.constraint_violations(constrained_probs)
            all_constraint_stats.append(constraint_stats)
        
        all_raw_logits.append(raw_logits.cpu())
        all_preds.append(constrained_probs.cpu())
        all_targets.append(labs.cpu())
    
    # Concatenate all results
    y_raw = torch.cat(all_raw_logits, dim=0)
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    
    # Compute comprehensive metrics
    metrics = compute_all_metrics(y_true, y_pred, pathologies, model.constraint_projection)
    
    # Compare with unconstrained predictions
    y_raw_probs = torch.sigmoid(y_raw)
    raw_metrics = compute_all_metrics(y_true, y_raw_probs, pathologies, None)
    
    # Add comparison metrics
    comparison_metrics = {}
    for key in ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro']:
        if key in metrics and key in raw_metrics:
            improvement = metrics[key] - raw_metrics[key]
            comparison_metrics[f'improvement_{key}'] = improvement
    
    return {
        'constrained': metrics,
        'unconstrained': raw_metrics,
        'improvement': comparison_metrics,
        'predictions': {
            'y_true': y_true,
            'y_pred_constrained': y_pred,
            'y_pred_raw': y_raw_probs
        }
    }


def create_visualizations(results, pathologies, output_dir):
    """Create comprehensive visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    y_true = results['predictions']['y_true'].numpy()
    y_pred = results['predictions']['y_pred_constrained'].numpy()
    y_raw = results['predictions']['y_pred_raw'].numpy()
    
    # 1. Per-class AUC comparison
    plt.figure(figsize=(12, 8))
    
    constrained_aucs = []
    unconstrained_aucs = []
    
    for i, pathology in enumerate(pathologies):
        key = f'auc_{pathology.lower()}'
        const_auc = results['constrained'].get(key, 0)
        raw_auc = results['unconstrained'].get(key, 0)
        constrained_aucs.append(const_auc)
        unconstrained_aucs.append(raw_auc)
    
    x = np.arange(len(pathologies))
    width = 0.35
    
    plt.bar(x - width/2, unconstrained_aucs, width, label='Unconstrained', alpha=0.8)
    plt.bar(x + width/2, constrained_aucs, width, label='Constrained', alpha=0.8)
    
    plt.xlabel('Pathologies')
    plt.ylabel('AUC Score')
    plt.title('Per-class AUC: Constrained vs Unconstrained')
    plt.xticks(x, pathologies, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Constraint violation heatmap
    if results['constrained'].get('implication_avg') is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Implication violations
        impl_violations = np.zeros((len(pathologies), len(pathologies)))
        for i, (a, b, _) in enumerate(MEDICAL_CONSTRAINTS['implications']):
            if a in pathologies and b in pathologies:
                ai, bi = pathologies.index(a), pathologies.index(b)
                # Compute actual violation rate
                violations = (y_pred[:, ai] > y_pred[:, bi]).mean()
                impl_violations[ai, bi] = violations
        
        sns.heatmap(impl_violations, annot=True, fmt='.3f', 
                   xticklabels=pathologies, yticklabels=pathologies,
                   ax=ax1, cmap='Reds')
        ax1.set_title('Implication Constraint Violations')
        
        # Exclusion violations
        excl_violations = np.zeros((len(pathologies), len(pathologies)))
        for i, (a, b, kappa) in enumerate(MEDICAL_CONSTRAINTS['exclusions']):
            if a in pathologies and b in pathologies:
                ai, bi = pathologies.index(a), pathologies.index(b)
                violations = ((y_pred[:, ai] + y_pred[:, bi]) > kappa).mean()
                excl_violations[ai, bi] = violations
                excl_violations[bi, ai] = violations  # Symmetric
        
        sns.heatmap(excl_violations, annot=True, fmt='.3f',
                   xticklabels=pathologies, yticklabels=pathologies, 
                   ax=ax2, cmap='Blues')
        ax2.set_title('Exclusion Constraint Violations')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'constraint_violations.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Prediction distribution comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Select top 6 most frequent pathologies for visualization
    pos_counts = y_true.sum(axis=0)
    top_indices = np.argsort(pos_counts)[-6:]
    
    for i, idx in enumerate(top_indices):
        pathology = pathologies[idx]
        
        # Plot prediction distributions
        pos_mask = y_true[:, idx] == 1
        neg_mask = y_true[:, idx] == 0
        
        if pos_mask.sum() > 0:
            axes[i].hist(y_pred[pos_mask, idx], bins=50, alpha=0.5, label='Positive', density=True)
            axes[i].hist(y_raw[pos_mask, idx], bins=50, alpha=0.5, label='Positive (Raw)', density=True, linestyle='--')
        
        if neg_mask.sum() > 0:
            axes[i].hist(y_pred[neg_mask, idx], bins=50, alpha=0.5, label='Negative', density=True)
            axes[i].hist(y_raw[neg_mask, idx], bins=50, alpha=0.5, label='Negative (Raw)', density=True, linestyle='--')
        
        axes[i].set_title(f'{pathology}')
        axes[i].set_xlabel('Prediction Probability')
        axes[i].set_ylabel('Density')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_detailed_results(results, pathologies):
    """Print detailed evaluation results."""
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print("-" * 40)
    constrained = results['constrained']
    unconstrained = results['unconstrained']
    improvement = results['improvement']
    
    metrics_to_show = ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro', 'precision_macro', 'recall_macro']
    
    print(f"{'Metric':<15} {'Unconstrained':<12} {'Constrained':<12} {'Improvement':<12}")
    print("-" * 55)
    
    for metric in metrics_to_show:
        unc_val = unconstrained.get(metric, 0)
        con_val = constrained.get(metric, 0)
        imp_val = improvement.get(f'improvement_{metric}', 0)
        print(f"{metric:<15} {unc_val:<12.4f} {con_val:<12.4f} {imp_val:<+12.4f}")
    
    # Per-class results for top pathologies
    print(f"\nPER-CLASS AUC (Top 10 by frequency):")
    print("-" * 60)
    
    y_true = results['predictions']['y_true'].numpy()
    pos_counts = y_true.sum(axis=0)
    top_indices = np.argsort(pos_counts)[-10:][::-1]
    
    print(f"{'Pathology':<20} {'Count':<8} {'Unconstrained':<12} {'Constrained':<12}")
    print("-" * 60)
    
    for idx in top_indices:
        pathology = pathologies[idx]
        count = int(pos_counts[idx])
        unc_auc = unconstrained.get(f'auc_{pathology.lower()}', 0)
        con_auc = constrained.get(f'auc_{pathology.lower()}', 0)
        print(f"{pathology:<20} {count:<8} {unc_auc:<12.4f} {con_auc:<12.4f}")
    
    # Constraint violations
    if constrained.get('implication_avg') is not None:
        print(f"\nCONSTRAINT VIOLATIONS:")
        print("-" * 30)
        print(f"Implication violations (avg): {constrained.get('implication_avg', 0):.6f}")
        print(f"Exclusion violations (avg):   {constrained.get('exclusion_avg', 0):.6f}")


def save_results(results, pathologies, output_dir):
    """Save detailed results to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    json_results = {
        'constrained_metrics': results['constrained'],
        'unconstrained_metrics': results['unconstrained'], 
        'improvement_metrics': results['improvement'],
        'pathologies': pathologies,
        'num_samples': results['predictions']['y_true'].shape[0]
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save predictions for further analysis
    predictions = {
        'y_true': results['predictions']['y_true'].numpy().tolist(),
        'y_pred_constrained': results['predictions']['y_pred_constrained'].numpy().tolist(),
        'y_pred_raw': results['predictions']['y_pred_raw'].numpy().tolist(),
        'pathologies': pathologies
    }
    
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Constraint-Projected Chest X-ray Classifier')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.data_root}")
    dataset = ChestXRay14Dataset(args.data_root, split=args.split, data_augmentation=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device, dataset.pathologies)
    
    # Run evaluation
    results = evaluate_model(model, loader, args.device, dataset.pathologies)
    
    # Print results
    print_detailed_results(results, dataset.pathologies)
    
    # Save results
    save_results(results, dataset.pathologies, args.output_dir)
    
    # Create visualizations
    if not args.no_viz:
        print(f"\nCreating visualizations...")
        create_visualizations(results, dataset.pathologies, args.output_dir)
        print(f"Visualizations saved to {args.output_dir}")
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()