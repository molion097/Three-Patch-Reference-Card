# Three-Patch Reference Card for Lightweight Color Relighting

Code and data for the SIVP paper *"Three-Patch Reference Card for Lightweight Color Relighting."*

## Overview

We relight a flat object's color from a source lighting condition to a target lighting condition using a compact three-patch ArUco reference card photographed with a smartphone camera.

The main model is a **factorized hybrid**: an analytical M3CB prior plus a learned MLP residual conditioned only on reference-patch features (no source object color in the learned component).

$$\hat{c}_{\text{tgt}} = \text{M3CB}(c_{\text{src}}, P_{\text{src}}, P_{\text{tgt}}) + \alpha \cdot \text{MLP}(\varphi(P_{\text{src}}, P_{\text{tgt}}))$$

## Repository Layout

```
src/colorrevision/              # installable Python package
scripts/                        # training, evaluation, figure generation
tests/                          # reproducibility and input-policy tests
configs/                        # dataset and model configuration
data/                           # real-world dataset (pickle)
manifests/                      # real-world dataset manifest (CSV)
splits/real_world_dataset_2/    # deterministic train/val/test splits (seed 42)
results/real_world_dataset_2/   # per-sample predictions for all 9 baselines
```

## Installation

Requires Python 3.10+ and conda.

```bash
conda create -n color python=3.10
conda activate color
pip install -e ".[dev]"
```

## Reproducing Table 1

```bash
# Train and evaluate the factorized hybrid on all four protocols
python scripts/train_model.py --protocol random_pair
python scripts/train_model.py --protocol object_holdout
python scripts/train_model.py --protocol light_holdout
python scripts/train_model.py --protocol object_and_light_holdout

# Collect metrics (mean ΔE₇₆, 95% bootstrap CI, Wilcoxon p-values)
python scripts/collect_metrics.py \
  --results-root results/real_world_dataset_2 \
  --output reports/metrics.csv
```

Pre-computed predictions for all nine baselines are in `results/real_world_dataset_2/`.

## Running Tests

```bash
pytest -q tests/
```

20 tests covering input-policy compliance (no source object color in MLP), light-holdout leakage checks, and split determinism.

## Evaluation Protocols

| Protocol | Description |
|---|---|
| `random_pair` | Random source-target lighting pairs; objects in train and test |
| `object_holdout` | Test objects unseen during training |
| `light_holdout` | Test lighting pairs unseen during training |
| `object_and_light_holdout` | Both objects and lighting pairs held out |

## Baselines

`copy_source`, `knn`, `diagonal`, `m3cb`, `channel_regression`, `white_patch`, `FC4` (adapted), `CFCC` (adapted), `factorized_hybrid`.

## Dataset

The real-world dataset (216 unique colors, 25 source-target lighting pairs, Samsung Galaxy A52s) is in `data/real_world_dataset_2.pkl`. The manifest CSV (`manifests/real_world_dataset_2.csv`) is a flat tabular export of the same data.

Synthetic pre-training data: 494,209 Blender-rendered patch pairs. Coming soon.

## Citation

Citation will be added upon acceptance.
