"""Reference-only transform-matrix MLP (sklearn MLPRegressor).

Input policy: reference-patch features only (36D). Source object RGB is NOT
part of the learned input.

The model learns to predict a 3x3 colour transform matrix from reference-patch
features. The learned transform is then blended with the M3CB analytical
transform using a validation-calibrated scale alpha:

    T_pred  = mlp.predict(scale(reference_patch_features))  # (9,) flattened
    T_blend = T_m3cb + alpha * (T_pred.reshape(3,3) - T_m3cb)
    pred    = clip(T_blend @ source_object_rgb, 0, 1)

Without calibration (alpha=1.0) the model collapses on unseen lighting pairs
(mean ΔE76 ≈ 30). With calibration (alpha selected on validation), it safely
falls back toward M3CB on hard holdout splits.

Architecture:
    StandardScaler → MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                                   solver='adam', alpha=1e-4,
                                   early_stopping=True, n_iter_no_change=50)
"""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor


def build_model(seed: int = 42, max_iter: int = 1000) -> MLPRegressor:
    """Return an unfitted sklearn MLPRegressor for transform-matrix prediction."""
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
