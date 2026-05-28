# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PyTorch project for SVHN digit classification using a LeNet CNN. The codebase has no external dependency declarations — dependencies are assumed available in the Python environment (PyTorch, torchvision, tqdm).

## Commands

Train a baseline model (10 epochs):
```
python train_SVHN.py
```

Train with data augmentation:
```
python train_SVHN.py --augmentation rotation
python train_SVHN.py --augmentation affine
python train_SVHN.py --augmentation colorjitter
```

Resume from a checkpoint:
```
python train_SVHN.py --resume ./outputs/checkpoint.pth.tar
```

Evaluate a trained model:
```
python eval_SVHN.py
python eval_SVHN.py --load ./outputs/model_best.pth.tar
```

Key arguments for `train_SVHN.py`: `--epochs` (default 10), `--lr` (default 0.001), `--batch-size` (default 32), `--augmentation` (`none`/`rotation`/`affine`/`colorjitter`), `--resume` (path to checkpoint).

## Architecture

- **`student_code.py`** — Core model definition and training/evaluation loops.
  - `LeNet(nn.Module)`: 3-channel input → conv1 (3→6, 5×5) → maxpool → conv2 (6→16, 5×5) → maxpool → flatten → fc1 (400→256) → fc2 (256→128) → fc3 (128→10). Returns both the output logits and a shape dict from intermediate layers.
  - `train_model()`: Standard training loop — forward, cross-entropy loss, backward, SGD step. Runs per-epoch, prints average loss.
  - `test_model()`: Evaluation loop computing per-class accuracy. Returns test accuracy as a float.
  - `count_model_params()`: Returns trainable parameter count in millions.

- **`train_SVHN.py`** — Training entry point. Sets up transforms, loads SVHN via the custom `SVHN` dataset wrapper, instantiates the model with SGD + CrossEntropyLoss, runs the epoch loop with checkpoint saving (keeps best model separately as `model_best.pth.tar`).

- **`eval_SVHN.py`** — Standalone evaluation. Loads a checkpoint and runs `test_model` once over the test set, printing accuracy and wall-clock time.

- **`dataloader.py`** — Thin wrapper around `torchvision.datasets.SVHN`. Accepts `"train"`, `"val"`, or `"test"` split (maps `"val"` to the official test split). Automatically downloads data to `./data/SVHN/`.

## Data augmentation

Four augmentation modes selectable via `--augmentation`:
- `none`: Only ToTensor + Normalize (SVHN channel means/std)
- `rotation`: RandomRotation(±10°)
- `affine`: RandomAffine (translate up to 10%)
- `colorjitter`: ColorJitter (brightness/contrast/saturation ±0.1)

Normalization uses dataset-wide values: mean `[0.4377, 0.4438, 0.4728]`, std `[0.1980, 0.2010, 0.1970]`.

## Outputs

Checkpoints and best model saved to `./outputs/`. SVHN dataset auto-downloaded to `./data/SVHN/`.
