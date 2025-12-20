from eurekamesh.core.metrics import MetricsTracker, calculate_upt


def test_metrics_tracker_consistency_and_rates():
    m = MetricsTracker()
    # Chunk 1: raw=10, valid=8, unique=5, dup=3
    m.add_chunk(raw_generated=10, valid_generated=8, unique=5, duplicates=3)
    # Chunk 2: raw=20, valid=15, unique=9, dup=6
    m.add_chunk(raw_generated=20, valid_generated=15, unique=9, duplicates=6)
    metrics = m.get_metrics()
    assert metrics['total_raw_generated'] == 30
    assert metrics['total_valid_generated'] == 23
    assert metrics['total_generated'] == 30  # legacy mapped to raw
    assert metrics['unique_accepted'] == 14
    assert metrics['duplicates_filtered'] == 9
    # Rates per chunk
    assert len(metrics['duplicate_rates']) == 2
    assert abs(metrics['duplicate_rates'][0] - (3/8)) < 1e-6
    assert abs(metrics['duplicate_rates'][1] - (6/15)) < 1e-6
    assert len(metrics['duplicate_rates_raw']) == 2
    assert abs(metrics['duplicate_rates_raw'][0] - (3/10)) < 1e-6
    assert abs(metrics['duplicate_rates_raw'][1] - (6/20)) < 1e-6
    # UPT uses total_generated (raw) by design for honest denominator
    upt = calculate_upt(metrics['total_generated'], metrics['unique_accepted'])
    assert abs(upt - (14/30)) < 1e-6


