from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000

from colorrevision.data.color_spaces import rgb_to_lab


def delta_e76_rgb(pred_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    pred_lab = rgb_to_lab(pred_rgb)
    target_lab = rgb_to_lab(target_rgb)
    return np.linalg.norm(pred_lab - target_lab, axis=-1)


def delta_e00_rgb(pred_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    pred_lab = rgb_to_lab(pred_rgb)
    target_lab = rgb_to_lab(target_rgb)
    return deltaE_ciede2000(pred_lab, target_lab)
