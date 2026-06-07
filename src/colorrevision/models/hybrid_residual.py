"""Factorized hybrid residual model (sklearn MLPRegressor).

Input policy: reference-patch features only (36D: source_patches, target_patches,
channel-wise ratios, channel-wise differences). Source object RGB is NOT part of
the learned input — it is applied deterministically after calibration.

Architecture:
    StandardScaler → MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                                   solver='adam', alpha=1e-4,
                                   early_stopping=True, n_iter_no_change=50)

Prediction at inference:
    base = m3cb(source_patches, target_patches, source_object_rgb)
    residual = mlp.predict(scale(reference_patch_features(source_patches, target_patches)))
    pred = clip(base + alpha * residual, 0, 1)

where alpha is selected on validation data from {0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}.
"""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor


def build_model(seed: int = 42, max_iter: int = 1000) -> MLPRegressor:
    """Return an unfitted sklearn MLPRegressor for hybrid residual training."""
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
