# NeuroForge AI
# Maintainer: Shahab Baloochi

import pytest

from neuroforge.quantization_report import summarize, plot_summary


def _row(seed, value):
    return {'distribution': 'normal', 'bits': 4, 'group_size': 32,
            'precision': 'fp32', 'seed': seed,
            'storage': {'effective_bits_per_weight': 5.},
            'total_output_error': {'relative_l2': value}}


def test_descriptive_summary_and_plot(tmp_path):
    result = summarize([_row(2, 0.3), _row(1, 0.1), _row(3, None)])
    assert len(result) == 1
    row = result[0]
    assert row['median_relative_l2'] == pytest.approx(0.2)
    assert row['iqr_relative_l2'] == pytest.approx(0.1)
    assert row['worst_relative_l2'] == 0.3
    assert row['valid_count'] == 2 and row['undefined_count'] == 1
    out = tmp_path / 'tradeoffs.png'
    plot_summary(result, out)
    assert out.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_undefined_and_invalid_summary():
    row = summarize([_row(1, None)])[0]
    assert row['median_relative_l2'] is None
    with pytest.raises(ValueError, match='Duplicate'):
        summarize([_row(1, 0.1), _row(1, 0.2)])
    with pytest.raises(ValueError, match='Invalid relative'):
        summarize([_row(1, -0.1)])
