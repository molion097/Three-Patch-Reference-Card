from __future__ import annotations

import numpy as np


def fit_diagonal_offset(source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Fit per-channel gain and offset with ridge regularization.

    A full 3x4 affine transform is underdetermined with only three patches, so this
    baseline uses the well-posed diagonal-plus-offset form.
    """
    source = np.asarray(source_patches_rgb, dtype=np.float64)
    target = np.asarray(target_patches_rgb, dtype=np.float64)
    params = []
    design = np.stack([source, np.ones_like(source)], axis=-1)
    for channel in range(3):
        x = design[:, channel, :]
        y = target[:, channel]
        lhs = x.T @ x + ridge * np.eye(2)
        rhs = x.T @ y
        params.append(np.linalg.solve(lhs, rhs))
    return np.asarray(params)


def predict(source_object_rgb: np.ndarray, source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    params = fit_diagonal_offset(source_patches_rgb, target_patches_rgb)
    source = np.asarray(source_object_rgb, dtype=np.float64)
    pred = source * params[:, 0] + params[:, 1]
    return np.clip(pred, 0.0, 1.0)
