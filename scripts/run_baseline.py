#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from colorrevision.baselines import affine_ls, channel_regression, copy_source, diagonal, full_matrix_ls, m3cb
from colorrevision.baselines.knn import KNNPatchTransform
from colorrevision.data.row_access import PATCH_COLUMNS, patches_from_row, rgb_from_row
from colorrevision.metrics.delta_e import delta_e00_rgb, delta_e76_rgb
from colorrevision.metrics.summary import summarize_errors
from colorrevision.stats.bootstrap import bootstrap_mean_ci


def _rgb(row: pd.Series, prefix: str) -> np.ndarray:
    return rgb_from_row(row, prefix)


def _patches(row: pd.Series, prefix: str) -> np.ndarray:
    return patches_from_row(row, prefix)


def _predict_method(method: str, row: pd.Series, knn_model: KNNPatchTransform | None = None) -> np.ndarray:
    source_object = _rgb(row, "source_object_rgb")
    source_patches = _patches(row, "source")
    target_patches = _patches(row, "target")
    if method == "copy_source":
        return copy_source.predict(source_object)
    if method == "diagonal":
        return diagonal.predict(source_object, source_patches, target_patches)
    if method == "m3cb":
        return m3cb.predict(source_object, source_patches, target_patches)
    if method == "channel_regression":
        return channel_regression.predict(source_object, source_patches, target_patches)
    if method == "affine_ls":
        return affine_ls.predict(source_object, source_patches, target_patches)
    if method == "full_matrix_ls":
        return full_matrix_ls.predict(source_object, source_patches, target_patches)
    if method == "knn":
        if knn_model is None:
            raise RuntimeError("k-NN baseline requires a fit model")
        return knn_model.predict(source_object, source_patches, target_patches)
    raise ValueError(f"Unknown method: {method}")


def _fit_knn(train_rows: pd.DataFrame) -> KNNPatchTransform:
    source_patches = np.asarray([_patches(row, "source") for _, row in train_rows.iterrows()])
    target_patches = np.asarray([_patches(row, "target") for _, row in train_rows.iterrows()])
    return KNNPatchTransform(k=1).fit(source_patches, target_patches)


def run_baseline(
    manifest_path: str,
    split_path: str,
    method: str,
    output_dir: str,
    split: str = "test",
    seed: int = 42,
) -> dict[str, float]:
    manifest = pd.read_csv(manifest_path)
    assignments = pd.read_csv(split_path)
    frame = manifest.merge(assignments, on="sample_id", how="inner")
    train_rows = frame[frame["split"] == "train"]
    eval_rows = frame[frame["split"] == split]
    if eval_rows.empty:
        raise ValueError(f"No rows found for split={split}")

    knn_model = _fit_knn(train_rows) if method == "knn" else None
    records = []
    for _, row in eval_rows.iterrows():
        pred = _predict_method(method, row, knn_model=knn_model)
        target = _rgb(row, "target_object_rgb")
        de76 = float(delta_e76_rgb(pred[None, :], target[None, :])[0])
        de00 = float(delta_e00_rgb(pred[None, :], target[None, :])[0])
        records.append(
            {
                "sample_id": row["sample_id"],
                "method": method,
                "split": split,
                "object_id": row["object_id"],
                "source_light_id": row["source_light_id"],
                "target_light_id": row["target_light_id"],
                "pred_r": pred[0],
                "pred_g": pred[1],
                "pred_b": pred[2],
                "target_r": target[0],
                "target_g": target[1],
                "target_b": target[2],
                "delta_e76": de76,
                "delta_e00": de00,
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame.from_records(records)
    predictions.to_csv(output_path / "predictions_test.csv", index=False)
    summary = summarize_errors(predictions["delta_e76"].to_numpy())
    summary.update(bootstrap_mean_ci(predictions["delta_e76"].to_numpy(), iterations=2000, seed=seed))
    summary["method"] = method
    summary["split"] = split
    with (output_path / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an analytical or trivial baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["copy_source", "diagonal", "m3cb", "channel_regression", "affine_ls", "full_matrix_ls", "knn"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_baseline(
        manifest_path=args.manifest,
        split_path=args.split_file,
        method=args.method,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
