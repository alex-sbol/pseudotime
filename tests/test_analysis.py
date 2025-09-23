import pandas as pd

from morphology_pipeline.analysis import compute_variability, detect_regions, records_to_dataframe
from morphology_pipeline.config import AnalysisConfig
from morphology_pipeline.data_models import CellRecord


def _sample_records():
    return [
        CellRecord(env_id="env1", image_index=1, cell_id=1, centroid_y=5, pseudotime=0.1, features={"area": 100.0}),
        CellRecord(env_id="env1", image_index=1, cell_id=2, centroid_y=15, pseudotime=0.3, features={"area": 150.0}),
        CellRecord(env_id="env2", image_index=1, cell_id=1, centroid_y=5, pseudotime=0.1, features={"area": 120.0}),
        CellRecord(env_id="env2", image_index=1, cell_id=2, centroid_y=15, pseudotime=0.3, features={"area": 180.0}),
    ]


def test_compute_variability():
    df = records_to_dataframe(_sample_records())
    stats_df = compute_variability(df, AnalysisConfig(), bins=5)
    assert not stats_df.empty
    regions = detect_regions(stats_df, AnalysisConfig())
    assert "is_driver" in regions.columns
