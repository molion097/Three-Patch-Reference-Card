from __future__ import annotations

import numpy as np

from colorrevision.features.transforms import apply_matrix, m3cb_matrix


class KNNPatchTransform:
    """Nearest-neighbor transform baseline using reference patch signatures."""

    def __init__(self, k: int = 1):
        self.k = k
        self.features: np.ndarray | None = None
        self.transforms: np.ndarray | None = None

    @staticmethod
    def _feature(source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(source_patches_rgb, dtype=np.float64).reshape(-1),
                np.asarray(target_patches_rgb, dtype=np.float64).reshape(-1),
            ]
        )

    def fit(self, source_patches_rgb: np.ndarray, target_patches_rgb: np.ndarray) -> "KNNPatchTransform":
        self.features = np.asarray(
            [self._feature(s, t) for s, t in zip(source_patches_rgb, target_patches_rgb)],
            dtype=np.float64,
        )
        self.transforms = np.asarray(
            [m3cb_matrix(s, t) for s, t in zip(source_patches_rgb, target_patches_rgb)],
            dtype=np.float64,
        )
        return self

    def predict(
        self,
        source_object_rgb: np.ndarray,
        source_patches_rgb: np.ndarray,
        target_patches_rgb: np.ndarray,
    ) -> np.ndarray:
        if self.features is None or self.transforms is None:
            raise RuntimeError("KNNPatchTransform must be fit before predict")
        feature = self._feature(source_patches_rgb, target_patches_rgb)
        distances = np.linalg.norm(self.features - feature, axis=1)
        neighbor_idx = np.argsort(distances)[: self.k]
        matrix = np.mean(self.transforms[neighbor_idx], axis=0)
        return apply_matrix(source_object_rgb, matrix)
