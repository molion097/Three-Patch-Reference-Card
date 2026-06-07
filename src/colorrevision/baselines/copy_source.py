from __future__ import annotations

import numpy as np


def predict(source_object_rgb: np.ndarray, **_: object) -> np.ndarray:
    return np.asarray(source_object_rgb, dtype=np.float64)
