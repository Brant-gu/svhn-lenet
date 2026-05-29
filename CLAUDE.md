# Project Notes

## Overview

PyTorch implementation of LeNet for SVHN digit classification.

```bash
pip install torch torchvision tqdm
```

## Dataset

[SVHN](http://ufldl.stanford.edu/housenumbers/) (Street View House Numbers) — 32×32 RGB images of house number digits (0–9), automatically downloaded to `./data/` on first run. Normalization: mean `[0.4377, 0.4438, 0.4728]`, std `[0.1980, 0.2010, 0.1970]`.

## Commands

```bash
# Baseline training (10 epochs)
python train_SVHN.py

# Training with data augmentation
python train_SVHN.py --augmentation rotation
python train_SVHN.py --augmentation affine
python train_SVHN.py --augmentation colorjitter

# Resume from checkpoint
python train_SVHN.py --resume ./outputs/checkpoint.pth.tar

# Evaluation
python eval_SVHN.py
python eval_SVHN.py --load ./outputs/model_best.pth.tar
```

Key arguments for `train_SVHN.py`: `--epochs` (default 10), `--lr` (default 0.001), `--batch-size` (default 32), `--augmentation` (`none`/`rotation`/`affine`/`colorjitter`), `--resume` (path to checkpoint).

## Architecture

- **`model.py`** — LeNet model, training loop, and evaluation loop.
  - `LeNet(nn.Module)`: 3×32×32 → conv1 (3→6, 5×5) → maxpool → conv2 (6→16, 5×5) → maxpool → flatten → fc1 (400→256) → fc2 (256→128) → fc3 (128→10). ~0.14M params.
  - `train_model()`: Standard training loop with cross-entropy loss and SGD.
  - `test_model()`: Evaluation loop returning test accuracy.

- **`train_SVHN.py`** — Training entry point with checkpoint saving (keeps best model as `model_best.pth.tar`).

- **`eval_SVHN.py`** — Standalone evaluation. Loads a checkpoint and reports accuracy and wall-clock time.

- **`dataloader.py`** — Thin wrapper around `torchvision.datasets.SVHN`. Accepts `"train"`, `"val"`, or `"test"` split (`"val"` maps to test split).

## Data augmentation

| Mode | Transform |
|------|-----------|
| `none` | ToTensor + Normalize only |
| `rotation` | RandomRotation(±10°) |
| `affine` | RandomAffine (translate up to 10%) |
| `colorjitter` | ColorJitter (brightness/contrast/saturation ±0.1) |

## Results

| Augmentation | Test Accuracy |
|-------------|--------------|
| None (baseline) | 88.64% |
| RandomRotation (±10°) | 88.94% |

## Outputs

Checkpoints saved to `./outputs/`. Dataset to `./data/`.
