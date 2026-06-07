"""Direct 18D object-free MLP — the reviewer's requested fix.

Input policy: reference patches ONLY. Source object RGB is excluded from the
learned input. This directly addresses the reviewer's complaint that the original
21D model was conditioned on observed objects, allowing it to interpolate object
identity rather than learning a general illuminant mapping.

Input (18D): source_patches (9D) + target_patches (9D)
Output (3D): predicted target_object_rgb (direct regression)

Architecture:
    StandardScaler → MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                                   solver='adam', alpha=1e-4,
                                   early_stopping=True, n_iter_no_change=50)
"""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor


def build_model(seed: int = 42, max_iter: int = 1000) -> MLPRegressor:
    """Return an unfitted sklearn MLPRegressor for direct-18D training."""
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
    )
