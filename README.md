# SVHN Digit Classification with LeNet

PyTorch implementation of LeNet for street view house numbers (SVHN) digit classification.

## Model Architecture

| Layer | Type | Input → Output | Kernel | Activation |
|-------|------|----------------|--------|------------|
| conv1 | Conv2d | 3×32×32 → 6×28×28 | 5×5 | ReLU |
| pool1 | MaxPool2d | 6×28×28 → 6×14×14 | 2×2, stride 2 | — |
| conv2 | Conv2d | 6×14×14 → 16×10×10 | 5×5 | ReLU |
| pool2 | MaxPool2d | 16×10×10 → 16×5×5 | 2×2, stride 2 | — |
| flatten | Flatten | 16×5×5 → 400 | — | — |
| fc1 | Linear | 400 → 256 | — | ReLU |
| fc2 | Linear | 256 → 128 | — | ReLU |
| fc3 | Linear | 128 → 10 | — | — |

Total trainable parameters: ~0.14M.

## Environment

- Python 3.x
- PyTorch ≥ 1.0
- torchvision
- tqdm

```bash
pip install torch torchvision tqdm
```

## Dataset

[SVHN](http://ufldl.stanford.edu/housenumbers/) (Street View House Numbers) — 32×32 RGB images of house number digits (0–9). The dataset is automatically downloaded to `./data/` on first run.

Normalization: mean = [0.4377, 0.4438, 0.4728], std = [0.1980, 0.2010, 0.1970].

## Quick Start

### Training

```bash
# Baseline (no augmentation)
python train_SVHN.py

# With data augmentation
python train_SVHN.py --augmentation rotation
python train_SVHN.py --augmentation affine
python train_SVHN.py --augmentation colorjitter

# Custom hyperparameters
python train_SVHN.py --epochs 20 --lr 0.01 --batch-size 64

# Resume from checkpoint
python train_SVHN.py --resume ./outputs/checkpoint.pth.tar
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 0.001 | Initial learning rate |
| `--batch-size` | 32 | Mini-batch size |
| `--augmentation` | none | Training augmentation: `none`, `rotation`, `affine`, `colorjitter` |
| `--resume` | — | Path to checkpoint for resuming training |

### Evaluation

```bash
python eval_SVHN.py
python eval_SVHN.py --load ./outputs/model_best.pth.tar
```

## Results

| Augmentation | Test Accuracy |
|-------------|--------------|
| None (baseline) | 88.64% |
| RandomRotation (±10°) | 88.94% |

Data augmentation (RandomRotation) yields a modest improvement over the baseline by exposing the model to varied orientations during training, which helps reduce overfitting and improves generalization.

## Project Structure

```
.
├── model.py          # LeNet model definition, train/test loops
├── train_SVHN.py     # Training script with checkpointing
├── eval_SVHN.py      # Standalone evaluation script
├── dataloader.py     # SVHN dataset wrapper (auto-download)
├── README.md
└── outputs/          # Saved checkpoints and best model
```
