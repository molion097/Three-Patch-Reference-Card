from __future__ import annotations

import numpy as np


def reference_patch_features(source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
    source = np.asarray(source_patches_rgb, dtype=np.float64).reshape(-1)
    target = np.asarray(target_patches_rgb, dtype=np.float64).reshape(-1)
    ratios = target / np.maximum(source, 1e-8)
    differences = target - source
    return np.concatenate([source, target, ratios, differences], axis=0)
