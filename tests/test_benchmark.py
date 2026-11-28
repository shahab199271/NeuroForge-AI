# NeuroForge AI
# Maintainer: Shahab Baloochi

import pytest

from neuroforge.benchmark import latency_summary


def test_nearest_rank_percentiles():
    result = latency_summary([4.0, 2.0, 1.0, 3.0, 5.0])
    assert result["p50_ms"] == 3.0
    assert result["p95_ms"] == 5.0
    assert result["requests_per_sec_sequential"] == pytest.approx(333.333, abs=0.001)


@pytest.mark.parametrize("data", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_latency_summary_rejects_invalid_data(data):
    with pytest.raises(ValueError):
        latency_summary(data)
