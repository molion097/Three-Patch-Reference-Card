import numpy as np

from colorrevision.metrics.delta_e import delta_e76_rgb


def test_delta_e76_zero_for_identical_rgb():
    rgb = np.array([[0.2, 0.4, 0.6], [1.0, 1.0, 1.0]])
    assert np.allclose(delta_e76_rgb(rgb, rgb), np.zeros(2))
