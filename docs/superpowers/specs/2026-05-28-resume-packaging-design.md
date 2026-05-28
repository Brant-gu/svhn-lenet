# Design: Resume-Ready Project Packaging

## Context

The SVHN LeNet project needs to be presented as a resume portfolio project targeting **CV engineer internship/new grad** roles. The goal is to optimize presentation only — no new features or experiments. The project should demonstrate solid fundamentals, experimental awareness, and clean code.

## Changes

### 1. Write README.md

A single comprehensive README covering:

- **Project title and one-liner**: SVHN digit classification using PyTorch LeNet
- **Model architecture**: Text/table describing layer-by-layer (input shape → Conv1 3→6 → Pool → Conv2 6→16 → Pool → Flatten → FC1 400→256 → FC2 256→128 → FC3 128→10), with output dimensions at each stage
- **Environment**: Python 3, PyTorch, torchvision, tqdm
- **Dataset**: SVHN (auto-downloaded to `./data/`), normalization stats
- **Quick start**: Train (baseline + augmentation variants), evaluate, key CLI args table
- **Experiment results**: Table showing baseline vs. each augmentation method (rotation, affine, colorjitter) with accuracy, plus brief analysis of why augmentation helps
- **Project structure**: File tree with one-line descriptions

### 2. Remove redundant files

- Delete `augmentation_analysis.txt` — its content (accuracy numbers + analysis) moves into README
- Delete `results.txt` — raw accuracy numbers folded into the results table in README

### 3. Files changed

| File | Action |
|------|--------|
| `README.md` | Create |
| `augmentation_analysis.txt` | Delete |
| `results.txt` | Delete |

No source code changes. The existing `model.py`, `train_SVHN.py`, `eval_SVHN.py`, `dataloader.py` remain as-is.

## Verification

1. Review README renders correctly (headings, tables, code blocks)
2. Confirm `augmentation_analysis.txt` and `results.txt` removed
3. Check no broken references to deleted files exist
