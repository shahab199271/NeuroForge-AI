"""Reproducible synthetic quantization error sweep (not a speed benchmark)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .kernels.error_analysis import analyze_quantization


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m', type=int, default=8)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--k', type=int, default=129)
    p.add_argument('--groups', type=int, nargs='+', default=[16, 32, 64, 128])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 29, 101])
    p.add_argument('--device', default='cpu')
    p.add_argument('--backend', choices=['torch', 'triton'], default='torch')
    p.add_argument('--output', type=Path, default=Path('outputs/quantization_errors.json'))
    args = p.parse_args()
    if min(args.m, args.n, args.k, *args.groups) < 1:
        p.error('Dimensions and groups must be positive')
    rows = []
    for seed in args.seeds:
        rng = torch.Generator().manual_seed(seed)
        a = torch.randn(args.m, args.k, generator=rng).to(args.device)
        w0 = torch.randn(args.n, args.k, generator=rng)
        for distribution in ('normal', 'outlier'):
            w = w0.clone()
            if distribution == 'outlier':
                w[:, ::31] *= 16
            w = w.to(args.device)
            for bits in (4, 8):
                for group in args.groups:
                    for precision in ('fp32', 'fp16', 'bf16'):
                        row = analyze_quantization(w, a, bits=bits, group_size=group,
                                                    precision=precision, backend=args.backend)
                        rows.append(dict(row, seed=seed, distribution=distribution))
    result = {'schema_version': 1, 'torch_version': str(torch.__version__),
              'cuda_version': torch.version.cuda,
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(f'Wrote {len(rows)} diagnostic rows to {args.output}')


if __name__ == '__main__':
    main()
