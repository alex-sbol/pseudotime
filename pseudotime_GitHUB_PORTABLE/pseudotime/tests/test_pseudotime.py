from morphology_pipeline.data_models import CellRecord
from morphology_pipeline.pseudotime import assign_pseudotime
from morphology_pipeline.period import PeriodEstimate


def test_assign_pseudotime():
    records = [
        CellRecord(env_id="env1", image_index=1, cell_id=1, centroid_y=10.0, pseudotime=0.0, features={}),
        CellRecord(env_id="env1", image_index=1, cell_id=2, centroid_y=35.0, pseudotime=0.0, features={}),
    ]
    period = PeriodEstimate(period=20.0, phase_offset=0.0, method="test")
    assign_pseudotime(records, period)
    assert records[0].pseudotime == 0.5
    assert 0 <= records[1].pseudotime < 1
