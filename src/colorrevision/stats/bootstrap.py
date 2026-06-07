from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(errors: np.ndarray, iterations: int = 10000, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        raise ValueError("Cannot bootstrap empty errors")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for idx in range(iterations):
        sample = rng.choice(errors, size=errors.size, replace=True)
        means[idx] = np.mean(sample)
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_ci_low": float(lower), "mean_ci_high": float(upper)}
