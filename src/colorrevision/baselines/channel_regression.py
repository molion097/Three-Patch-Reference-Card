from __future__ import annotations

import numpy as np

from colorrevision.features.transforms import apply_channel_regression, channel_regression_params


def predict(source_object_rgb: np.ndarray, source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    params = channel_regression_params(source_patches_rgb, target_patches_rgb)
    return apply_channel_regression(source_object_rgb, params)
