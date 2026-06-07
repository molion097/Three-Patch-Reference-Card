import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from colorrevision.baselines.full_matrix_ls import fit_matrix


def _load_train_model_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("train_model_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(source_object):
    return {
        "source_patch_r_r": 255.0,
        "source_patch_r_g": 0.0,
        "source_patch_r_b": 0.0,
        "source_patch_g_r": 0.0,
        "source_patch_g_g": 255.0,
        "source_patch_g_b": 0.0,
        "source_patch_b_r": 0.0,
        "source_patch_b_g": 0.0,
        "source_patch_b_b": 255.0,
        "target_patch_r_r": 204.0,
        "target_patch_r_g": 25.5,
        "target_patch_r_b": 0.0,
        "target_patch_g_r": 0.0,
        "target_patch_g_g": 229.5,
        "target_patch_g_b": 25.5,
        "target_patch_b_r": 51.0,
        "target_patch_b_g": 0.0,
        "target_patch_b_b": 178.5,
        "source_object_rgb_r": source_object[0],
        "source_object_rgb_g": source_object[1],
        "source_object_rgb_b": source_object[2],
        "target_object_rgb_r": 0.3,
        "target_object_rgb_g": 0.4,
        "target_object_rgb_b": 0.5,
    }


def test_transform_matrix_build_xy_is_reference_patch_only():
    train_model = _load_train_model_module()
    rows = pd.DataFrame([_row((51.0, 102.0, 153.0)), _row((178.5, 25.5, 76.5))])
    x, y = train_model._build_xy(rows, "transform_matrix")

    assert x.shape == (2, 36)
    assert y.shape == (2, 9)
    assert np.allclose(x[0], x[1])
    assert np.allclose(y[0], y[1])


def test_transform_matrix_build_xy_targets_full_matrix_ls():
    train_model = _load_train_model_module()
    rows = pd.DataFrame([_row((51.0, 102.0, 153.0))])
    _, y = train_model._build_xy(rows, "transform_matrix")
    expected = fit_matrix(
        np.eye(3),
        np.array(
            [
                [0.8, 0.1, 0.0],
                [0.0, 0.9, 0.1],
                [0.2, 0.0, 0.7],
            ]
        ),
    ).reshape(-1)
    assert np.allclose(y[0], expected)


def test_transform_matrix_predict_rows_keeps_source_object_out_of_model_input():
    train_model = _load_train_model_module()
    rows = pd.DataFrame([_row((51.0, 102.0, 153.0)), _row((178.5, 25.5, 76.5))])
    x, _ = train_model._build_xy(rows, "transform_matrix")
    scaler = StandardScaler().fit(x)

    class RecordingModel:
        def __init__(self):
            self.inputs = []

        def predict(self, x_input):
            self.inputs.append(np.asarray(x_input))
            return np.tile(np.eye(3).reshape(1, -1), (len(x_input), 1))

    model = RecordingModel()
    records = train_model._predict_rows(
        "transform_matrix",
        model,
        scaler,
        rows.assign(
            split="test",
            sample_id=["a", "b"],
            object_id=["o1", "o2"],
            source_light_id=["s1", "s1"],
            target_light_id=["t1", "t1"],
        ),
    )

    assert all(model_input.shape == (1, 36) for model_input in model.inputs)
    assert not np.allclose([records[0]["pred_r"], records[0]["pred_g"], records[0]["pred_b"]], [records[1]["pred_r"], records[1]["pred_g"], records[1]["pred_b"]])


def test_transform_matrix_scale_zero_falls_back_to_m3cb():
    train_model = _load_train_model_module()
    rows = pd.DataFrame([_row((51.0, 102.0, 153.0))]).assign(
        split="test",
        sample_id=["a"],
        object_id=["o1"],
        source_light_id=["s1"],
        target_light_id=["t1"],
    )
    x, _ = train_model._build_xy(rows, "transform_matrix")
    scaler = StandardScaler().fit(x)

    class BadModel:
        def predict(self, x_input):
            return np.tile((np.eye(3) * 100).reshape(1, -1), (len(x_input), 1))

    record = train_model._predict_rows("transform_matrix", BadModel(), scaler, rows, residual_scale=0.0)[0]
    m3cb = train_model._m3cb_pred(rows.iloc[0])
    assert np.allclose([record["pred_r"], record["pred_g"], record["pred_b"]], m3cb)


def test_transform_matrix_calibration_selects_best_scale():
    train_model = _load_train_model_module()
    rows = pd.DataFrame([_row((51.0, 102.0, 153.0))]).assign(
        split="val",
        sample_id=["a"],
        object_id=["o1"],
        source_light_id=["s1"],
        target_light_id=["t1"],
    )
    x, _ = train_model._build_xy(rows, "transform_matrix")
    scaler = StandardScaler().fit(x)

    class IdentityModel:
        def predict(self, x_input):
            return np.tile(np.eye(3).reshape(1, -1), (len(x_input), 1))

    scale = train_model._calibrate_residual_scale(
        "transform_matrix",
        IdentityModel(),
        scaler,
        rows,
        candidates=(0.0, 1.0),
    )
    assert scale == 0.0
