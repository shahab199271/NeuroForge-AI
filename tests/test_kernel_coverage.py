# NeuroForge AI
# Maintainer: Shahab Baloochi

import pytest

from neuroforge.kernel_coverage import CPU_MODULES, coverage_summary


def _report():
    row = {'num_statements': 8, 'covered_lines': 7,
           'num_branches': 2, 'covered_branches': 1}
    files = {f'src/neuroforge/kernels/{name}': {'summary': dict(row)}
             for name in CPU_MODULES}
    files['src/neuroforge/kernels/triton_ops.py'] = {
        'summary': dict(num_statements=10, covered_lines=0,
                        num_branches=0, covered_branches=0)}
    totals = {k: sum(f['summary'][k] for f in files.values()) for k in row}
    return {'meta': {'branch_coverage': True}, 'files': files, 'totals': totals}


def test_weighted_coverage():
    result = coverage_summary(_report())
    assert result['cpu_facing_combined_percent'] == 80
    assert result['whole_kernel_combined_percent'] == 72


def test_reject_missing_or_nonbranch_report():
    report = _report()
    report['files'].pop('src/neuroforge/kernels/ops.py')
    with pytest.raises(ValueError, match='Missing'):
        coverage_summary(report)
    report = _report()
    report['meta']['branch_coverage'] = False
    with pytest.raises(ValueError, match='Branch'):
        coverage_summary(report)
