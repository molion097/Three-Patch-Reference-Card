from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab


def as_float_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.size and np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = as_float_rgb(rgb)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = as_float_rgb(rgb)
    original_shape = rgb.shape
    flat = rgb.reshape(-1, 1, 3)
    lab = rgb2lab(flat).reshape(-1, 3)
    return lab.reshape(original_shape)
