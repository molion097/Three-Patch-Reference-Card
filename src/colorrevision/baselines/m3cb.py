from __future__ import annotations

import numpy as np

from colorrevision.features.transforms import apply_matrix, m3cb_matrix


def predict(source_object_rgb: np.ndarray, source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    matrix = m3cb_matrix(source_patches_rgb, target_patches_rgb)
    return apply_matrix(source_object_rgb, matrix)
