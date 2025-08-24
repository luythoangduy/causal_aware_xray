# Causal / Constrained Chest X-Ray Classification

Implements a differentiable constraint projection (Dykstra) layer for multi-label Chest X-ray classification (ChestX-ray14) integrating medical knowledge (implications & exclusions) to enforce anatomical/physiological plausibility.

## Features
- ConstraintProjection layer (implication & exclusion linear inequality handling)
- Integration with TorchXRayVision pretrained backbones (DenseNet, ResNet, EfficientNet, ViT)
- Flexible application: integrated (in-graph) or post-process
- Composite loss (supervised + constraint penalty + consistency)
- Rich evaluation: AUC (macro/micro/per-class), sensitivity, specificity, MCC, constraint violation rate

## Dataset Structure
Point `--data-root` to folder containing `nih_chestxray_14` directory (images + CSV metadata). Example:
```
C:/chest-xray/nih_chestxray_14/
  |-- images_001/            # Image files batch 1
  |-- images_002/            # Image files batch 2
  |-- images_003/            # Image files batch 3
  |-- ...                    # More image subdirectories
  |-- Data_Entry_2017.csv
  |-- train_val_list.txt
  |-- test_list.txt
```

## Quickstart

### Installation
```bash
pip install -e .
```

### Training

**Setup Weights & Biases (optional):**
1. Copy template config:
   ```bash
   cp configs/train_template.yaml configs/train_private.yaml
   ```
2. Edit `configs/train_private.yaml` and add your wandb API key and entity
3. Or set environment variable:
   ```bash
   export WANDB_API_KEY="your-wandb-api-key"
   ```

**Basic training:**
```bash
python -m src.training.train --data-root C:/chest-xray/nih_chestxray_14 --config configs/train.yaml
```

**With Weights & Biases logging:**
```bash
python -m src.training.train \
    --data-root C:/chest-xray/nih_chestxray_14 \
    --config configs/train_private.yaml \
    --output-dir outputs/experiment_1
```

**Disable W&B logging:**
```bash
python -m src.training.train \
    --data-root C:/chest-xray/nih_chestxray_14 \
    --config configs/train.yaml \
    --no-wandb
```

Resume from checkpoint:
```bash
python -m src.training.train \
    --data-root C:/chest-xray/nih_chestxray_14 \
    --config configs/train.yaml \
    --resume outputs/checkpoint_epoch_10.pth
```

### Evaluation
Evaluate trained model:
```bash
python -m src.evaluation.evaluate \
    --checkpoint outputs/best_model.pth \
    --data-root C:/chest-xray/nih_chestxray_14 \
    --output-dir evaluation_results
```

### Testing
Run test suite:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_integration.py -v
```

## License
MIT
