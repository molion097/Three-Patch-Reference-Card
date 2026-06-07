import numpy as np

from colorrevision.external.image_illuminants import (
    gray_edge_illuminant,
    gray_world_illuminant,
    shades_of_gray_illuminant,
    white_patch_illuminant,
)


def test_image_illuminant_estimators_return_unit_rgb():
    image = np.ones((8, 8, 3), dtype=float)
    image[..., 0] = 0.5
    for fn in [gray_world_illuminant, white_patch_illuminant, shades_of_gray_illuminant, gray_edge_illuminant]:
        illum = fn(image)
        assert illum.shape == (3,)
        assert np.isclose(np.linalg.norm(illum), 1.0)
