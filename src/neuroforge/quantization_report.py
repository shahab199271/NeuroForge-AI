# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Descriptive cross-seed summaries and error/storage trade-off plots."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import iqr


def summarize(rows: list[dict]) -> list[dict]:
    """Summarize matched configurations; undefined metrics remain explicit."""
    groups = defaultdict(list)
    for row in rows:
        key = (row['distribution'], row['bits'], row['group_size'], row['precision'])
        groups[key].append(row)
    result = []
    for (distribution, bits, size, precision), group in sorted(groups.items()):
        seeds = [r['seed'] for r in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate seeds for a configuration')
        storage = [r['storage']['effective_bits_per_weight'] for r in group]
        if not np.all(np.isfinite(storage)) or len(set(storage)) != 1 or storage[0] <= 0:
            raise ValueError('Inconsistent effective storage')
        values = [r['total_output_error']['relative_l2'] for r in group]
        valid = np.array([v for v in values if v is not None], dtype=np.float64)
        if not np.all(np.isfinite(valid)) or np.any(valid < 0):
            raise ValueError('Invalid relative error')
        result.append({
            'distribution': distribution, 'bits': bits, 'group_size': size,
            'precision': precision, 'seeds': sorted(seeds),
            'effective_bits_per_weight': storage[0],
            'valid_count': int(valid.size), 'undefined_count': len(values)-int(valid.size),
            'median_relative_l2': float(np.median(valid)) if valid.size else None,
            'iqr_relative_l2': float(iqr(valid)) if valid.size else None,
            'worst_relative_l2': float(valid.max()) if valid.size else None,
        })
    return result


def plot_summary(summary: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    distributions = sorted({r['distribution'] for r in summary})
    if not distributions:
        raise ValueError('No configurations to plot')
    fig, axes = plt.subplots(1, len(distributions), figsize=(6*len(distributions), 4.8),
                             squeeze=False, layout='constrained')
    for ax, dist in zip(axes[0], distributions):
        for bits in (4, 8):
            for precision, marker in [('fp32', 'o'), ('fp16', 's'), ('bf16', '^')]:
                points = [r for r in summary if r['distribution'] == dist
                          and r['bits'] == bits and r['precision'] == precision
                          and r['median_relative_l2'] is not None]
                points.sort(key=lambda r: r['effective_bits_per_weight'])
                if points:
                    ax.plot([r['effective_bits_per_weight'] for r in points],
                            [r['median_relative_l2'] for r in points],
                            marker=marker, label=f'INT{bits} / {precision.upper()}')
        ax.set(title=dist.capitalize(), xlabel='Effective bits per weight',
               ylabel='Median relative L2 output error')
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle('Synthetic quantization diagnostics (descriptive across seeds)')
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/validation'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    summary = summarize(data['rows'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'quantization-summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    plot_summary(summary, args.output_dir / 'quantization-tradeoffs.png')
    print(f'Summarized {len(summary)} matched configurations')


if __name__ == '__main__':
    main()
