from __future__ import annotations

import numpy as np

from colorrevision.features.transforms import diagonal_gain


def predict(source_object_rgb: np.ndarray, source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    gain = diagonal_gain(source_patches_rgb, target_patches_rgb)
    return np.clip(np.asarray(source_object_rgb, dtype=np.float64) * gain, 0.0, 1.0)
