from __future__ import annotations

import numpy as np


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "n": float(errors.size),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
        "p95": float(np.percentile(errors, 95)),
        "fail_gt_2": float(np.mean(errors > 2.0)),
        "fail_gt_5": float(np.mean(errors > 5.0)),
        "fail_gt_10": float(np.mean(errors > 10.0)),
    }
