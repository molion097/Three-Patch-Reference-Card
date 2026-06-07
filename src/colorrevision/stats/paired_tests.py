from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a_errors: np.ndarray, b_errors: np.ndarray) -> dict[str, float]:
    a = np.asarray(a_errors, dtype=np.float64)
    b = np.asarray(b_errors, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must match, got {a.shape} and {b.shape}")
    result = wilcoxon(a, b, zero_method="zsplit")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
