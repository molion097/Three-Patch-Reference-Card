from __future__ import annotations

import numpy as np
import pandas as pd


PATCH_COLUMNS = {
    "source": [
        ["source_patch_r_r", "source_patch_r_g", "source_patch_r_b"],
        ["source_patch_g_r", "source_patch_g_g", "source_patch_g_b"],
        ["source_patch_b_r", "source_patch_b_g", "source_patch_b_b"],
    ],
    "target": [
        ["target_patch_r_r", "target_patch_r_g", "target_patch_r_b"],
        ["target_patch_g_r", "target_patch_g_g", "target_patch_g_b"],
        ["target_patch_b_r", "target_patch_b_g", "target_patch_b_b"],
    ],
}


def rgb_from_row(row: pd.Series, prefix: str) -> np.ndarray:
    return row[[f"{prefix}_r", f"{prefix}_g", f"{prefix}_b"]].to_numpy(dtype=np.float64) / 255.0


def patches_from_row(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray(
        [row[columns].to_numpy(dtype=np.float64) / 255.0 for columns in PATCH_COLUMNS[prefix]],
        dtype=np.float64,
    )
