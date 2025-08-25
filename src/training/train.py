from __future__ import annotations
import argparse
import os
import torch
import yaml
import wandb
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

from src.data.chestxray14 import ChestXRay14Dataset, PATHOLOGIES
from src.models.constraint_projection import ConstraintProjection
from src.models.classifier import ConstrainedChestXRayClassifier
from src.training.losses import ConstrainedLoss
from src.evaluation.metrics import compute_all_metrics


MEDICAL_CONSTRAINTS = {
    # Hierarchical implications - based on anatomical/physiological relationships
    'implications': [
        # Pneumonia pathway - specific to general
        ('Pneumonia', 'Lung Opacity', 0.0),           # Pneumonia is a type of lung opacity
        ('Pneumonia', 'Consolidation', -0.1),         # Pneumonia often causes consolidation 
        ('Consolidation', 'Lung Opacity', 0.0),       # Consolidation is lung opacity
        
        # Lung pathology hierarchy
        ('Consolidation', 'Infiltration', 0.0),       # Consolidation implies infiltration
        ('Mass', 'Nodule', 0.0),                      # Mass is large nodule
        ('Atelectasis', 'Lung Opacity', 0.0),         # Atelectasis causes opacity
        ('Lesion', 'Lung Opacity', 0.1),              # Lesions generally cause opacity
        
        # Cardiac-related implications
        ('Cardiomegaly', 'Enlarged Cardiom.', 0.0),   # Direct relationship
        ('Enlarged Cardiom.', 'Edema', -0.1),         # Cardiac enlargement can cause edema
        ('Cardiomegaly', 'Edema', -0.2),              # Less direct but common association
        
        # Pleural pathology
        ('Pleural Effusion', 'Pleural Other', 0.0),   # Effusion is pleural abnormality
        ('Pneumothorax', 'Pleural Other', 0.05),      # Pneumothorax affects pleura
        
        # Support devices implications
        ('Support Devices', 'No Finding', -0.11),     # If devices present, less likely "no finding"
        
        # Fracture implications
        ('Fracture', 'Support Devices', 0.05),        # Fractures may need support
    ],
    
    # Medical exclusions - mutually incompatible conditions
    'exclusions': [
        # Primary exclusions - anatomically/physiologically incompatible
        ('Pneumothorax', 'Pleural Effusion', 1.4),    # Cannot have both simultaneously
        ('Emphysema', 'Fibrosis', 1.2),               # Different pathophysiology
        ('Pneumothorax', 'Consolidation', 1.1),       # Mechanically incompatible
        
        # Moderate exclusions - rarely co-occur
        ('Atelectasis', 'Emphysema', 1.0),            # Opposite lung mechanics
        ('Hernia', 'Pneumothorax', 1.0),              # Different anatomical regions
        ('Edema', 'Pneumothorax', 0.9),               # Fluid vs air pathology
        
        # Mild exclusions - can co-occur but unusual
        ('Mass', 'Pneumonia', 0.8),                   # Different etiologies
        ('Nodule', 'Edema', 0.7),                     # Different pathophysiology
        ('Fibrosis', 'Consolidation', 0.7),           # Chronic vs acute
        
        # No Finding exclusions
        ('No Finding', 'Pneumonia', 0.11),            # Cannot be both normal and abnormal
        ('No Finding', 'Consolidation', 0.11),
        ('No Finding', 'Atelectasis', 0.11),
        ('No Finding', 'Cardiomegaly', 0.11),
        ('No Finding', 'Pleural Effusion', 0.11),
        ('No Finding', 'Mass', 0.11),
        ('No Finding', 'Nodule', 0.11),
        ('No Finding', 'Pneumothorax', 0.11),
        ('No Finding', 'Infiltration', 0.11),
        ('No Finding', 'Edema', 0.11),
        ('No Finding', 'Emphysema', 0.11),
        ('No Finding', 'Fibrosis', 0.11),
        ('No Finding', 'Pleural Thickening', 0.11),
        ('No Finding', 'Hernia', 0.11),
    ],
    
    # Hierarchical structure mapping for reference
    'hierarchy': {
        'root': ['Model'],
        'Model': [
            'No Finding', 'Enlarged Cardiom.', 'Cardiomegaly', 
            'Support Devices', 'Lung Opacity', 'Pleural Other', 
            'Pleural Effusion', 'Pneumothorax', 'Fracture', 'Edema'
        ],
        'Lung Opacity': ['Consolidation', 'Pneumonia', 'Atelectasis', 'Lesion'],
        'Lesion': [],
        'Consolidation': [],
        'Pneumonia': [],
        'Atelectasis': [],
        'Pleural Other': [],
        'Pleural Effusion': [],
        'Pneumothorax': [],
        'Enlarged Cardiom.': [],
        'Cardiomegaly': [],
        'Support Devices': [],
        'Fracture': [],
        'Edema': [],
        'No Finding': []
    },
}



def setup_wandb(cfg, args):
    """Initialize Weights & Biases logging."""
    wandb_cfg = cfg.get('wandb', {})
    
    if not wandb_cfg.get('project'):
        print("Warning: No wandb project specified, logging disabled")
        return False
    
    # Set API key if provided in config
    api_key = wandb_cfg.get('api_key')
    if api_key:
        os.environ['WANDB_API_KEY'] = api_key
        print("Using wandb API key from config")
    elif os.getenv('WANDB_API_KEY'):
        print("Using wandb API key from environment variable")
    else:
        print("Warning: No wandb API key found in config or environment variable")
        print("You may need to run 'wandb login' or set WANDB_API_KEY")
    
    try:
        wandb.init(
            project=wandb_cfg.get('project', 'causal-xray-loss'),
            entity=wandb_cfg.get('entity'),
            name=wandb_cfg.get('name', f"run_{int(time.time())}"),
            tags=wandb_cfg.get('tags', []),
            notes=wandb_cfg.get('notes', ''),
            config={
                **cfg,
                'data_root': args.data_root,
                'device': args.device,
            }
        )
        print(f"Wandb initialized successfully. Project: {wandb_cfg.get('project')}")
        return True
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        print("Training will continue without wandb logging")
        return False


def build_constraint_layer(pathologies, cfg):
    """Build constraint projection layer."""
    label_to_idx = {p: i for i, p in enumerate(pathologies)}
    implications = [(label_to_idx[a], label_to_idx[b], tau) for a, b, tau in MEDICAL_CONSTRAINTS['implications']]
    exclusions = [(label_to_idx[a], label_to_idx[b], kappa) for a, b, kappa in MEDICAL_CONSTRAINTS['exclusions']]
    
    max_iter = cfg.get('constraint_iterations', 20)
    return ConstraintProjection(implications=implications, exclusions=exclusions, max_iter=max_iter)


def create_scheduler(optimizer, cfg, total_steps):
    """Create learning rate scheduler."""
    scheduler_type = cfg.get('scheduler', 'cosine')
    warmup_epochs = cfg.get('warmup_epochs', 0)
    warmup_steps = warmup_epochs * (total_steps // cfg.get('epochs', 10))
    
    if scheduler_type == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    else:
        main_scheduler = None
    
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        if main_scheduler:
            return SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])
        return warmup_scheduler
    
    return main_scheduler


def train_epoch(model, loss_fn, loader, optimizer, scheduler, device, epoch, log_wandb=True):
    """Train for one epoch with comprehensive logging."""
    model.train()
    logs = {}
    step_logs = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        imgs = batch['img'].to(device)
        labs = batch['lab'].to(device)
        
        # Forward pass - get raw logits
        raw_logits = model(imgs, apply_constraints=False)
        
        # Calculate constraint violations on RAW probabilities (before projection)
        with torch.no_grad():
            raw_probs = torch.sigmoid(raw_logits)
            constraint_stats = model.constraint_projection.constraint_violations(raw_probs) if model.constraint_projection else {}
        
        # Get constrained predictions (apply projection)
        constrained_probs = model.project(raw_logits)
        
        # Compute loss using pre-projection constraint stats
        losses = loss_fn(constrained_probs, raw_logits, labs, constraint_stats)
        
        # Backward pass
        optimizer.zero_grad()
        losses['loss'].backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Log metrics
        step_metrics = {
            'train/loss': losses['loss'].item(),
            'train/loss_primary': losses['primary'].item(),
            'train/loss_constraint': losses['constraint_penalty'].item(),
            'train/loss_consistency': losses['consistency'].item(),
            'train/grad_norm': grad_norm.item(),
            'train/lr': optimizer.param_groups[0]['lr'],
        }
        
        # Add constraint violations (both pre and post projection for monitoring)
        for k, v in constraint_stats.items():
            step_metrics[f'train/constraint_pre_{k}'] = v
            
        # Optionally add post-projection violations for comparison
        if model.constraint_projection:
            with torch.no_grad():
                post_stats = model.constraint_projection.constraint_violations(constrained_probs)
                for k, v in post_stats.items():
                    step_metrics[f'train/constraint_post_{k}'] = v
        
        step_logs.append(step_metrics)
        
        # Log to wandb every 50 steps
        if log_wandb and batch_idx % 50 == 0:
            wandb.log(step_metrics, step=epoch * len(loader) + batch_idx)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['loss'].item():.4f}",
            'constraint': f"{losses['constraint_penalty'].item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # Accumulate for epoch average
        for k, v in step_metrics.items():
            logs.setdefault(k, []).append(v)
    
    # Compute epoch averages
    epoch_metrics = {k: np.mean(v) for k, v in logs.items()}
    
    return epoch_metrics


@torch.no_grad()
def evaluate(model, loader, device, pathologies, epoch, split='val', log_wandb=True):
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_constraint_stats = []
    
    pbar = tqdm(loader, desc=f'Eval {split}')
    for batch in pbar:
        imgs = batch['img'].to(device)
        labs = batch['lab'].to(device)
        
        # Get predictions
        constrained_probs = model(imgs)  # Uses integrated mode
        
        # Get constraint violations
        if model.constraint_projection:
            constraint_stats = model.constraint_projection.constraint_violations(constrained_probs)
            all_constraint_stats.append(constraint_stats)
        
        all_preds.append(constrained_probs.cpu())
        all_targets.append(labs.cpu())
    
    # Concatenate all predictions
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, pathologies, model.constraint_projection)
    
    # Add split prefix
    eval_metrics = {f'{split}/{k}': v for k, v in metrics.items()}
    
    # Log key metrics
    print(f"{split.upper()} - Epoch {epoch}:")
    print(f"  AUC Macro: {metrics.get('auc_macro', 0):.4f}")
    print(f"  AUC Micro: {metrics.get('auc_micro', 0):.4f}")
    print(f"  F1 Macro:  {metrics.get('f1_macro', 0):.4f}")
    
    if log_wandb:
        wandb.log(eval_metrics, step=epoch)
    
    return eval_metrics, y_pred, y_true


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Constraint-Projected Chest X-ray Classifier')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Config file path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup logging
    log_wandb = not args.no_wandb and setup_wandb(cfg, args)
    
    # Create datasets
    train_ds = ChestXRay14Dataset(args.data_root, split='train')
    val_ds = ChestXRay14Dataset(args.data_root, split='test')  # Using test as validation
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.get('batch_size', 16),
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build model
    constraint_layer = build_constraint_layer(train_ds.pathologies, cfg)
    model = ConstrainedChestXRayClassifier(
        backbone=cfg.get('backbone', 'densenet121'),
        pretrained=False,
        num_classes=len(train_ds.pathologies),
        constraint_projection=constraint_layer,
        projection_mode=cfg.get('projection_mode', 'integrated')
    )
    model.to(args.device)
    
    # Create loss function
    loss_fn = ConstrainedLoss(
        constraint_weight=cfg.get('constraint_weight', 1.0),
        consistency_weight=cfg.get('consistency_weight', 0.5)
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.get('learning_rate', 1e-4),
        weight_decay=cfg.get('weight_decay', 1e-5)
    )
    
    # Create scheduler
    total_steps = len(train_loader) * cfg.get('epochs', 10)
    scheduler = create_scheduler(optimizer, cfg, total_steps)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_auc = 0.0
    epochs = cfg.get('epochs', 10)
    eval_every = cfg.get('eval_every', 1)
    save_every = cfg.get('save_every', 5)
    
    for epoch in range(start_epoch, epochs):
        # Training
        train_metrics = train_epoch(model, loss_fn, train_loader, optimizer, scheduler, args.device, epoch, log_wandb)
        
        if log_wandb:
            wandb.log({k: v for k, v in train_metrics.items() if not k.endswith('lr')}, step=epoch)
        
        # Evaluation
        if epoch % eval_every == 0:
            val_metrics, val_preds, val_targets = evaluate(model, val_loader, args.device, train_ds.pathologies, epoch, 'val', log_wandb)
            
            # Save best model
            current_auc = val_metrics.get('val/auc_macro', 0)
            if current_auc > best_auc:
                best_auc = current_auc
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    os.path.join(args.output_dir, 'best_model.pth')
                )
        
        # Save periodic checkpoint
        if epoch % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    final_metrics, _, _ = evaluate(model, val_loader, args.device, train_ds.pathologies, epochs, 'test', log_wandb)
    
    if log_wandb:
        wandb.finish()
    
    print(f"Training completed. Best validation AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
