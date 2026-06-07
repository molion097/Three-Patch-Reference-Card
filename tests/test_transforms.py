import numpy as np

from colorrevision.features.transforms import apply_matrix, diagonal_gain, m3cb_matrix


def test_m3cb_maps_reference_patches_exactly_on_toy_example():
    source = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = np.array(
        [
            [0.8, 0.1, 0.0],
            [0.0, 0.9, 0.1],
            [0.2, 0.0, 0.7],
        ]
    )
    matrix = m3cb_matrix(source, target)
    predicted = apply_matrix(source, matrix, clip=False)
    assert np.allclose(predicted, target)


def test_diagonal_gain_identity_for_same_patches():
    patches = np.array(
        [
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7],
            [0.8, 0.7, 0.6],
        ]
    )
    assert np.allclose(diagonal_gain(patches, patches), np.ones(3))
