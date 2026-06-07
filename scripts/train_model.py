#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from colorrevision.data.row_access import patches_from_row, rgb_from_row
from colorrevision.features.patch_features import reference_patch_features
from colorrevision.metrics.delta_e import delta_e00_rgb, delta_e76_rgb
from colorrevision.metrics.summary import summarize_errors
from colorrevision.stats.bootstrap import bootstrap_mean_ci
from colorrevision.baselines import full_matrix_ls, m3cb
from colorrevision.features.transforms import apply_matrix
from colorrevision.models import direct_18d as _direct_18d_mod
from colorrevision.models import direct_21d as _direct_21d_mod
from colorrevision.models import hybrid_residual as _hybrid_mod
from colorrevision.models import transform_mlp as _transform_mod


def _feature_reference(row: pd.Series) -> np.ndarray:
    return reference_patch_features(patches_from_row(row, "source"), patches_from_row(row, "target"))


def _feature_direct_18d(row: pd.Series) -> np.ndarray:
    return np.concatenate(
        [
            patches_from_row(row, "source").reshape(-1),
            patches_from_row(row, "target").reshape(-1),
        ]
    )


def _feature_direct_21d(row: pd.Series) -> np.ndarray:
    return np.concatenate(
        [
            patches_from_row(row, "source").reshape(-1),
            patches_from_row(row, "target").reshape(-1),
            rgb_from_row(row, "source_object_rgb"),
        ]
    )


def _m3cb_pred(row: pd.Series) -> np.ndarray:
    return m3cb.predict(
        rgb_from_row(row, "source_object_rgb"),
        patches_from_row(row, "source"),
        patches_from_row(row, "target"),
    )


def _build_xy(rows: pd.DataFrame, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    if model_name in {"direct_18d", "direct_21d"}:
        feat_fn = _feature_direct_18d if model_name == "direct_18d" else _feature_direct_21d
        x = np.asarray([feat_fn(row) for _, row in rows.iterrows()])
        y = np.asarray([rgb_from_row(row, "target_object_rgb") for _, row in rows.iterrows()])
        return x, y
    if model_name == "transform_diagonal":
        x = np.asarray([_feature_reference(row) for _, row in rows.iterrows()])
        y = []
        for _, row in rows.iterrows():
            source = rgb_from_row(row, "source_object_rgb")
            target = rgb_from_row(row, "target_object_rgb")
            y.append(target / np.maximum(source, 1e-4))
        return x, np.asarray(y)
    if model_name == "transform_matrix":
        x = np.asarray([_feature_reference(row) for _, row in rows.iterrows()])
        y = []
        for _, row in rows.iterrows():
            matrix = full_matrix_ls.fit_matrix(
                patches_from_row(row, "source"),
                patches_from_row(row, "target"),
            )
            y.append(matrix.reshape(-1))
        return x, np.asarray(y)
    if model_name == "hybrid_residual_rgb":
        x = np.asarray([_feature_reference(row) for _, row in rows.iterrows()])
        y = []
        for _, row in rows.iterrows():
            target = rgb_from_row(row, "target_object_rgb")
            y.append(target - _m3cb_pred(row))
        return x, np.asarray(y)
    raise ValueError(f"Unknown model: {model_name}")


def _predict_rows(
    model_name: str,
    model: MLPRegressor,
    scaler: StandardScaler,
    rows: pd.DataFrame,
    residual_scale: float = 1.0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        if model_name == "direct_18d":
            x = _feature_direct_18d(row)[None, :]
            pred = model.predict(scaler.transform(x))[0]
        elif model_name == "direct_21d":
            x = _feature_direct_21d(row)[None, :]
            pred = model.predict(scaler.transform(x))[0]
        elif model_name == "transform_diagonal":
            x = _feature_reference(row)[None, :]
            gains = model.predict(scaler.transform(x))[0]
            pred = rgb_from_row(row, "source_object_rgb") * gains
        elif model_name == "transform_matrix":
            x = _feature_reference(row)[None, :]
            matrix = model.predict(scaler.transform(x))[0].reshape(3, 3)
            learned_pred = apply_matrix(rgb_from_row(row, "source_object_rgb"), matrix)
            pred = _m3cb_pred(row) + residual_scale * (learned_pred - _m3cb_pred(row))
        elif model_name == "hybrid_residual_rgb":
            x = _feature_reference(row)[None, :]
            residual = model.predict(scaler.transform(x))[0]
            pred = _m3cb_pred(row) + residual_scale * residual
        else:
            raise ValueError(f"Unknown model: {model_name}")
        pred = np.clip(pred, 0.0, 1.0)
        target = rgb_from_row(row, "target_object_rgb")
        records.append(
            {
                "sample_id": row["sample_id"],
                "method": model_name,
                "split": row["split"],
                "object_id": row["object_id"],
                "source_light_id": row["source_light_id"],
                "target_light_id": row["target_light_id"],
                "pred_r": pred[0],
                "pred_g": pred[1],
                "pred_b": pred[2],
                "target_r": target[0],
                "target_g": target[1],
                "target_b": target[2],
                "delta_e76": float(delta_e76_rgb(pred[None, :], target[None, :])[0]),
                "delta_e00": float(delta_e00_rgb(pred[None, :], target[None, :])[0]),
            }
        )
    return records


def _mean_de76_for_scale(
    model_name: str,
    model: MLPRegressor,
    scaler: StandardScaler,
    rows: pd.DataFrame,
    residual_scale: float,
) -> float:
    predictions = _predict_rows(model_name, model, scaler, rows, residual_scale=residual_scale)
    return float(np.mean([record["delta_e76"] for record in predictions]))


def _calibrate_residual_scale(
    model_name: str,
    model: MLPRegressor,
    scaler: StandardScaler,
    val_rows: pd.DataFrame,
    candidates: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
) -> float:
    if model_name not in {"hybrid_residual_rgb", "transform_matrix"}:
        return 1.0
    if val_rows.empty:
        return 0.0
    scored = [
        (scale, _mean_de76_for_scale(model_name, model, scaler, val_rows, residual_scale=scale))
        for scale in candidates
    ]
    valid = [(s, v) for s, v in scored if not (v != v)]  # drop NaN
    return min(valid, key=lambda item: item[1])[0] if valid else 0.0


def train_and_evaluate(
    manifest_path: str,
    split_file: str,
    model_name: str,
    output_dir: str,
    seed: int = 42,
    max_iter: int = 1000,
    residual_scale: float = 1.0,
    calibrate_residual_scale: bool = False,
) -> dict[str, float]:
    manifest = pd.read_csv(manifest_path)
    assignments = pd.read_csv(split_file)
    frame = manifest.merge(assignments, on="sample_id", how="inner")
    train_rows = frame[frame["split"] == "train"]
    val_rows = frame[frame["split"] == "val"]
    test_rows = frame[frame["split"] == "test"]

    x_train, y_train = _build_xy(train_rows, model_name)
    scaler = StandardScaler().fit(x_train)
    _model_builders = {
        "direct_18d": _direct_18d_mod.build_model,
        "direct_21d": _direct_21d_mod.build_model,
        "transform_diagonal": _hybrid_mod.build_model,
        "transform_matrix": _transform_mod.build_model,
        "hybrid_residual_rgb": _hybrid_mod.build_model,
    }
    model = _model_builders[model_name](seed=seed, max_iter=max_iter)
    model.fit(scaler.transform(x_train), y_train)

    if calibrate_residual_scale:
        residual_scale = _calibrate_residual_scale(model_name, model, scaler, val_rows)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "scaler": scaler, "model_name": model_name}, handle)

    predictions = pd.DataFrame.from_records(_predict_rows(model_name, model, scaler, test_rows, residual_scale=residual_scale))
    predictions.to_csv(output_path / "predictions_test.csv", index=False)
    summary = summarize_errors(predictions["delta_e76"].to_numpy())
    summary.update(bootstrap_mean_ci(predictions["delta_e76"].to_numpy(), iterations=2000, seed=seed))
    summary.update(
        {
            "method": model_name,
            "split": "test",
            "train_n": int(len(train_rows)),
            "val_n": int(len(val_rows)),
            "test_n": int(len(test_rows)),
            "n_iter": int(model.n_iter_),
            "residual_scale": float(residual_scale) if model_name in {"hybrid_residual_rgb", "transform_matrix"} else None,
            "learned_input_policy": "includes_source_object" if model_name == "direct_21d" else "reference_patches_only",
        }
    )
    with (output_path / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural relighting model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["direct_18d", "direct_21d", "transform_diagonal", "transform_matrix", "hybrid_residual_rgb"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--calibrate-residual-scale", action="store_true")
    args = parser.parse_args()
    summary = train_and_evaluate(
        manifest_path=args.manifest,
        split_file=args.split_file,
        model_name=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        max_iter=args.max_iter,
        residual_scale=args.residual_scale,
        calibrate_residual_scale=args.calibrate_residual_scale,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
