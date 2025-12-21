import pytest
from eurekamesh.core.metrics import MetricsTracker

def test_metrics_upt_duplicate_rate():
    m = MetricsTracker()
    # valid_generated = unique + duplicates for each chunk
    m.add_chunk(raw_generated=10, valid_generated=10, unique=7, duplicates=3)
    m.add_chunk(raw_generated=8, valid_generated=8, unique=6, duplicates=2)
    m.add_chunk(raw_generated=12, valid_generated=12, unique=8, duplicates=4)
    metrics = m.get_metrics()
    assert metrics['total_generated'] == 30
    assert metrics['unique_accepted'] == 21
    assert metrics['duplicates_filtered'] == 9
    upt = metrics['unique_accepted'] / metrics['total_generated']
    dup_rate = metrics['duplicates_filtered'] / metrics['total_generated']
    assert abs(upt - 0.7) < 1e-6
    assert abs(dup_rate - 0.3) < 1e-6
