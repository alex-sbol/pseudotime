import numpy as np
import pandas as pd

from morphology_pipeline.pseudolocation_demo import build_target_bins, rank_cells_for_targets


def test_build_target_bins_for_three_steps():
    assert np.array_equal(build_target_bins(17, 3), np.array([0, 8, 16]))


def test_build_target_bins_for_four_steps():
    assert np.array_equal(build_target_bins(17, 4), np.array([0, 5, 11, 16]))


def test_rank_cells_for_targets_uses_unique_rows_per_unit():
    df = pd.DataFrame(
        {
            "unique_key": ["a0", "a1", "b0", "b1", "c0", "c1"],
            "center_bin": [0, 0, 5, 5, 10, 10],
            "area": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "path_background": ["bg"] * 6,
            "path_dapi": ["dapi"] * 6,
        }
    )

    ranked = rank_cells_for_targets(df, target_bins=[0, 5, 10], n_units=2)

    assert len(ranked) == 6
    assert ranked["unique_key"].nunique() == 6
    assert ranked["target_bin"].tolist() == [0, 5, 10, 0, 5, 10]
    assert ranked["center_bin"].tolist() == [0, 5, 10, 0, 5, 10]
