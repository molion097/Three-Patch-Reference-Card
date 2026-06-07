"""
Pre-train the hybrid residual MLP on synthetic data, then fine-tune on real.

Pipeline
--------
1. Pre-train StandardScaler + MLPRegressor on synthetic_with_patches.csv
   (train split, random_pair protocol)
2. Warm-start fine-tune on real_world_dataset_2 (train split, random_pair)
   by initialising weights from the pre-trained model
3. Calibrate residual_alpha on real validation fold
4. Evaluate on real test fold (all 4 protocols)
5. Save model → results/real_world_dataset_2/<protocol>/hybrid_pretrained/model.pkl
"""
from __future__ import annotations

import pickle
import pathlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from colorrevision.data.row_access import patches_from_row, rgb_from_row
from colorrevision.features.patch_features import reference_patch_features
from colorrevision.baselines import m3cb
from colorrevision.metrics.delta_e import delta_e76_rgb
from colorrevision.metrics.summary import summarize_errors
from colorrevision.stats.bootstrap import bootstrap_mean_ci

REPO           = pathlib.Path(r"D:\colorrevision\newcode")
SYNTH_MANIFEST = REPO / "manifests" / "synthetic_with_patches.csv"
REAL_MANIFEST  = REPO / "manifests" / "real_world_dataset_2.csv"

PROTOCOLS = [
    "random_pair",
    "object_holdout",
    "light_holdout",
    "object_and_light_holdout",
]
ALPHA_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


# ── data helpers (same pattern as train_model.py) ─────────────────────────────
def _m3cb_pred(row: pd.Series) -> np.ndarray:
    return m3cb.predict(
        rgb_from_row(row, "source_object_rgb"),
        patches_from_row(row, "source"),
        patches_from_row(row, "target"),
    )


def build_xy(rows: pd.DataFrame):
    """Build (X, y, to_, base) for hybrid_residual model.

    Follows the same per-row logic as train_model.py:_build_xy so that
    patches_from_row / rgb_from_row / reference_patch_features are called
    identically to the main training pipeline.
    Both real and synthetic manifests store patch values as 0-255 integers.
    """
    X_list, y_list, base_list, to_list = [], [], [], []
    for _, row in rows.iterrows():
        sp   = patches_from_row(row, "source")            # (3,3) float [0,1]
        tp   = patches_from_row(row, "target")            # (3,3)
        so   = rgb_from_row(row, "source_object_rgb")     # (3,) float [0,1]
        to_  = rgb_from_row(row, "target_object_rgb")     # (3,)
        feat = reference_patch_features(sp, tp)           # (36,)
        base = _m3cb_pred(row)                            # (3,)
        X_list.append(feat)
        y_list.append(to_ - base)
        base_list.append(base)
        to_list.append(to_)
    return (np.array(X_list), np.array(y_list),
            np.array(to_list), np.array(base_list))


def load_split(manifest_path, split_path, split_name: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    splits   = pd.read_csv(split_path)
    merged   = manifest.merge(splits, on="sample_id", how="inner")
    return merged[merged["split"] == split_name].reset_index(drop=True)


# ── step 1: pre-train on synthetic random_pair ────────────────────────────────
synth_split = REPO / "splits" / "synthetic" / "random_pair" / "seed_42.csv"

print("=== Step 1: Pre-training on synthetic data ===")
train_syn = load_split(SYNTH_MANIFEST, synth_split, "train")
print(f"Synthetic train: {len(train_syn):,}")

print("Building synthetic features...")
X_syn_tr, y_syn_tr, *_ = build_xy(train_syn)

scaler_syn = StandardScaler().fit(X_syn_tr)
X_syn_tr_s = scaler_syn.transform(X_syn_tr)

mlp_pre = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=30,
)
mlp_pre.fit(X_syn_tr_s, y_syn_tr)
print(f"Pre-train converged in {mlp_pre.n_iter_} iterations")


# ── step 2: fine-tune on real data (warm start per protocol) ─────────────────
print("\n=== Step 2: Fine-tuning on real-world data ===")

for protocol in PROTOCOLS:
    real_split = REPO / "splits" / "real_world_dataset_2" / protocol / "seed_42.csv"
    if not real_split.exists():
        print(f"  [{protocol}] split file not found, skipping")
        continue

    train_real = load_split(REAL_MANIFEST, real_split, "train")
    val_real   = load_split(REAL_MANIFEST, real_split, "val")
    test_real  = load_split(REAL_MANIFEST, real_split, "test")
    print(f"\n  [{protocol}] train={len(train_real)} val={len(val_real)} test={len(test_real)}")

    X_tr,  y_tr,  _,      base_tr  = build_xy(train_real)
    X_val, y_val, to_val, base_val = build_xy(val_real)
    X_te,  y_te,  to_te,  base_te  = build_xy(test_real)

    # Fit a new scaler on real training data
    scaler_real = StandardScaler().fit(X_tr)

    # Manually transfer pre-trained weights, then fine-tune
    # Bootstrap fit (2 iterations, no early stopping) to allocate weight arrays.
    # Use full X_tr so validation_fraction is not too small.
    mlp_ft = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=5e-4,
        max_iter=2,
        random_state=42,
        early_stopping=False,
        warm_start=False,
    )
    mlp_ft.fit(scaler_real.transform(X_tr), y_tr)

    # Overwrite weights with pre-trained values.
    mlp_ft.coefs_      = [c.copy() for c in mlp_pre.coefs_]
    mlp_ft.intercepts_ = [b.copy() for b in mlp_pre.intercepts_]

    # Reset Adam optimizer and ALL early-stopping state so fine-tune starts clean.
    # The bootstrap fit (early_stopping=False) leaves validation_scores_=None;
    # warm_start skips re-init, so we must reset everything manually.
    mlp_ft._optimizer             = None
    mlp_ft.best_loss_             = np.inf
    mlp_ft._no_improvement_count  = 0
    mlp_ft.loss_curve_            = []
    mlp_ft.validation_scores_     = []
    mlp_ft.best_validation_score_ = -np.inf

    # Fine-tune with proper settings.
    mlp_ft.early_stopping      = True
    mlp_ft.validation_fraction = 0.10
    mlp_ft.n_iter_no_change    = 30
    mlp_ft.warm_start          = True
    mlp_ft.max_iter            = 300
    mlp_ft.fit(scaler_real.transform(X_tr), y_tr)
    print(f"    Fine-tune converged in {mlp_ft.n_iter_} iterations")

    # Calibrate alpha on validation
    best_alpha, best_de = 0.0, float("inf")
    for alpha in ALPHA_GRID:
        pred_val = np.clip(
            base_val + alpha * mlp_ft.predict(scaler_real.transform(X_val)), 0, 1
        )
        de = delta_e76_rgb(pred_val, to_val).mean()
        if de < best_de:
            best_de, best_alpha = de, alpha
    print(f"    Best alpha={best_alpha}  val ΔE={best_de:.4f}")

    # Evaluate on test
    pred_te = np.clip(
        base_te + best_alpha * mlp_ft.predict(scaler_real.transform(X_te)), 0, 1
    )
    de_te = delta_e76_rgb(pred_te, to_te)
    summary = summarize_errors(de_te)
    summary.update(bootstrap_mean_ci(de_te, iterations=2000, seed=0))
    summary["method"]          = "hybrid_pretrained"
    summary["protocol"]        = protocol
    summary["residual_scale"]  = best_alpha
    summary["n_iter_pretrain"] = mlp_pre.n_iter_
    summary["n_iter_finetune"] = mlp_ft.n_iter_
    print(f"    Test ΔE76={summary['mean']:.4f} [{summary['mean_ci_low']:.4f}, {summary['mean_ci_high']:.4f}]")

    # Save
    out_dir = REPO / "results" / "real_world_dataset_2" / protocol / "hybrid_pretrained"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump({"model": mlp_ft, "scaler": scaler_real,
                     "model_name": "hybrid_pretrained",
                     "residual_scale": best_alpha}, f)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"    Saved → {out_dir}")

print("\n=== Done ===")
