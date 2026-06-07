import numpy as np
import pytest

from colorrevision.baselines.full_matrix_ls import fit_matrix, predict
from colorrevision.features.transforms import apply_matrix


def test_full_matrix_ls_recovers_known_transform():
    source = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.2],
            [0.2, 0.1, 0.7],
        ]
    )
    true_matrix = np.array([[0.8, 0.1, 0.0], [0.0, 0.9, 0.1], [0.2, 0.0, 0.7]])
    target = apply_matrix(source, true_matrix, clip=False)
    matrix = fit_matrix(source, target, ridge=0.0)
    assert np.allclose(apply_matrix(source, matrix, clip=False), target)
    assert np.allclose(matrix, true_matrix)


def test_full_matrix_ls_predicts_object_with_known_transform():
    source_patches = np.eye(3)
    target_patches = np.array(
        [
            [0.7, 0.1, 0.0],
            [0.0, 0.8, 0.1],
            [0.1, 0.0, 0.9],
        ]
    )
    source_object = np.array([0.2, 0.4, 0.6])
    expected = apply_matrix(source_object, fit_matrix(source_patches, target_patches, ridge=0.0))
    actual = predict(source_object, source_patches, target_patches)
    assert np.allclose(actual, expected, atol=1e-5)


def test_full_matrix_ls_rejects_bad_patch_shape():
    with pytest.raises(ValueError):
        fit_matrix(np.ones((2, 3)), np.ones((3, 3)))
