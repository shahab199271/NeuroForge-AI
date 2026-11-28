# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Report coverage honestly, retaining the full package as a separate gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CPU_MODULES = frozenset({
    '__init__.py', 'ops.py', 'advanced.py', 'quantization.py',
    'group_quantization.py', 'int4.py', 'int4_gemm.py',
    'mixed_precision.py', 'error_analysis.py',
})


def coverage_summary(report: dict[str, Any]) -> dict[str, float]:
    """Use covered lines plus covered branches, not an average of percentages."""
    found: dict[str, dict[str, Any]] = {}
    for name, info in report['files'].items():
        path = Path(name)
        if path.name in CPU_MODULES and path.parent.name == 'kernels':
            if path.name in found:
                raise ValueError(f'Duplicate module: {path.name}')
            found[path.name] = info['summary']
    if set(found) != CPU_MODULES:
        raise ValueError(f'Missing CPU-facing modules: {CPU_MODULES - set(found)}')
    if not report['meta']['branch_coverage']:
        raise ValueError('Branch coverage is required')

    def percent(items: list[dict[str, Any]]) -> float:
        total = sum(int(s['num_statements']) + int(s['num_branches']) for s in items)
        covered = sum(int(s['covered_lines']) + int(s['covered_branches']) for s in items)
        if total <= 0 or not 0 <= covered <= total:
            raise ValueError('Invalid coverage totals')
        return 100 * covered / total

    return {'cpu_facing_combined_percent': percent(list(found.values())),
            'whole_kernel_combined_percent': percent([report['totals']])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--cpu-min', type=float, default=80.)
    p.add_argument('--whole-min', type=float, default=55.)
    args = p.parse_args()
    if not all(0 <= v <= 100 for v in (args.cpu_min, args.whole_min)):
        p.error('Thresholds must be between 0 and 100')
    result = coverage_summary(json.loads(args.report.read_text(encoding='utf-8')))
    print(json.dumps(result, indent=2))
    if (result['cpu_facing_combined_percent'] < args.cpu_min
            or result['whole_kernel_combined_percent'] < args.whole_min):
        raise SystemExit('Coverage regression below configured floor')


if __name__ == '__main__':
    main()
