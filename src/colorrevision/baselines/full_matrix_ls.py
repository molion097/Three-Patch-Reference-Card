from __future__ import annotations

import numpy as np

from colorrevision.features.transforms import apply_matrix


def fit_matrix(source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Fit a ridge-regularized 3x3 source-to-target RGB transform.

    With three RGB reference patches this is a square patch-transform estimate,
    not an overdetermined multi-patch fit. At ``ridge=0`` and full rank it is
    equivalent to the pseudoinverse-style full matrix used by M3CB.
    """
    source = np.asarray(source_patches_rgb, dtype=np.float64)
    target = np.asarray(target_patches_rgb, dtype=np.float64)
    if source.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError(f"Expected source/target patch shapes (3, 3), got {source.shape} and {target.shape}")
    lhs = source.T @ source + ridge * np.eye(3)
    rhs = source.T @ target
    return np.linalg.solve(lhs, rhs).T


def predict(source_object_rgb: np.ndarray, source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    matrix = fit_matrix(source_patches_rgb, target_patches_rgb)
    return apply_matrix(source_object_rgb, matrix)
