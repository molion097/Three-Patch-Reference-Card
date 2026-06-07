import pandas as pd

from colorrevision.data.row_access import patches_from_row, rgb_from_row


def test_row_access_rgb_and_patch_shapes():
    row = pd.Series(
        {
            "source_object_rgb_r": 10,
            "source_object_rgb_g": 20,
            "source_object_rgb_b": 30,
            "source_patch_r_r": 1,
            "source_patch_r_g": 2,
            "source_patch_r_b": 3,
            "source_patch_g_r": 4,
            "source_patch_g_g": 5,
            "source_patch_g_b": 6,
            "source_patch_b_r": 7,
            "source_patch_b_g": 8,
            "source_patch_b_b": 9,
        }
    )
    assert rgb_from_row(row, "source_object_rgb").shape == (3,)
    assert patches_from_row(row, "source").shape == (3, 3)
