"""Direct 21D object-conditioned MLP — ABLATION ONLY.

Input policy: includes source object RGB (3D). This is the input design the
reviewer objected to: the model is conditioned on observed objects and therefore
interpolates object identity rather than learning a general illuminant mapping.
It should NOT be reported as the main proposed method.

Input (21D): source_patches (9D) + target_patches (9D) + source_object_rgb (3D)
Output (3D): predicted target_object_rgb (direct regression)

Architecture:
    StandardScaler → MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                                   solver='adam', alpha=1e-4,
                                   early_stopping=True, n_iter_no_change=50)

Known failure mode: collapses on light-holdout (mean ΔE76 ≈ 44) because
the model memorises object-colour/illuminant correlations seen during training.
"""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor


def build_model(seed: int = 42, max_iter: int = 1000) -> MLPRegressor:
    """Return an unfitted sklearn MLPRegressor for direct-21D ablation training."""
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
