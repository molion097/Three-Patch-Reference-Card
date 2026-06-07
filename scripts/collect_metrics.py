#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_BOOTSTRAP_N = 2000
_BOOTSTRAP_SEED = 0


def _bootstrap_ci(values: np.ndarray, n: int = _BOOTSTRAP_N, seed: int = _BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _de00_stats(pred_path: Path) -> dict:
    try:
        df = pd.read_csv(pred_path, usecols=["delta_e00"])
        vals = df["delta_e00"].dropna().to_numpy()
        if len(vals) == 0:
            return {}
        lo, hi = _bootstrap_ci(vals)
        return {
            "de00_mean": float(vals.mean()),
            "de00_median": float(np.median(vals)),
            "de00_p95": float(np.percentile(vals, 95)),
            "de00_mean_ci_low": lo,
            "de00_mean_ci_high": hi,
        }
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect metrics_summary.json files into one CSV.")
    parser.add_argument("--results-root", default="newcode/results/real_world_dataset_2")
    parser.add_argument("--output", default="newcode/reports/real_world_dataset_2_metrics.csv")
    args = parser.parse_args()

    records = []
    for path in sorted(Path(args.results_root).glob("**/metrics_summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        parts = path.relative_to(args.results_root).parts
        if len(parts) >= 3:
            record["protocol"] = parts[0]
            record["run_name"] = parts[1]
        record["metrics_path"] = str(path)
        pred_path = path.parent / "predictions_test.csv"
        record.update(_de00_stats(pred_path))
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No metrics_summary.json files under {args.results_root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).sort_values(["protocol", "mean"]).to_csv(output, index=False)
    print(f"Wrote {len(records)} metric rows to {output}")


if __name__ == "__main__":
    main()
