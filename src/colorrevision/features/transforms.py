from __future__ import annotations

import numpy as np


def patches_to_matrix(patches_rgb: np.ndarray) -> np.ndarray:
    """Return 3x3 matrix whose columns are the three reference patch RGB vectors."""
    patches = np.asarray(patches_rgb, dtype=np.float64)
    if patches.shape != (3, 3):
        raise ValueError(f"Expected patches shape (3, 3), got {patches.shape}")
    return patches.T


def m3cb_matrix(source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    source = patches_to_matrix(source_patches_rgb)
    target = patches_to_matrix(target_patches_rgb)
    return target @ np.linalg.pinv(source)


def apply_matrix(rgb: np.ndarray, matrix: np.ndarray, clip: bool = True) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    pred = rgb @ np.asarray(matrix, dtype=np.float64).T
    return np.clip(pred, 0.0, 1.0) if clip else pred


def diagonal_gain(source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    source_mean = np.mean(np.asarray(source_patches_rgb, dtype=np.float64), axis=0)
    target_mean = np.mean(np.asarray(target_patches_rgb, dtype=np.float64), axis=0)
    return target_mean / np.maximum(source_mean, eps)


def channel_regression_params(source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    source = np.asarray(source_patches_rgb, dtype=np.float64)
    target = np.asarray(target_patches_rgb, dtype=np.float64)
    params = []
    for channel in range(3):
        x = source[:, channel]
        y = target[:, channel]
        a, b = np.polyfit(x, y, deg=1)
        params.append((a, b))
    return np.asarray(params, dtype=np.float64)


def apply_channel_regression(rgb: np.ndarray, params: np.ndarray, clip: bool = True) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    pred = rgb * params[:, 0] + params[:, 1]
    return np.clip(pred, 0.0, 1.0) if clip else pred
