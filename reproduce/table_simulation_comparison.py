#!/usr/bin/env python3
"""Comparison tables with explicit sample size and exposure distribution."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]  # this checkout's reproduce/ and neuralplsi
from reproduce.result_io import load_results, compute_summaries, curves, true_g


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/simulation')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    records = load_results(args.input)
    df = compute_summaries(records)
    if df.empty:
        raise ValueError('No estimates to compare')
    extra = []
    for rec in records:
        common = {k: rec['_'+k] for k in ('model', 'n', 'g_fn', 'outcome', 'x_dist')}
        subset = df
        for k, v in common.items():
            subset = subset[subset[k] == v]
        for method, group in subset.groupby('inference_type'):
            for block in ('beta', 'gamma'):
                bias = group.loc[group.param.str.startswith(block+'_'), 'bias']
                extra.append(dict(common, inference_type=method, param=block+'_SqBias', bias=float(np.sum(bias**2))))
        grid, values = curves(rec)
        if grid is not None:
            truth = true_g(grid, rec['_g_fn'])
            if rec['_outcome'] == 'cox':
                truth -= float(true_g(0., rec['_g_fn']))
            mse = ((values-truth)**2).mean(axis=1)
            extra.append(dict(common, inference_type='Point estimate', param='g_MSE', bias=float(mse.mean()),
                              emp_sd=float(mse.std(ddof=1)) if len(mse)>1 else np.nan))
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    out = Path(args.out or args.input)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/'comparison_metrics.csv', index=False)
    for (n, distribution), group in df.groupby(['n', 'x_dist']):
        table = group.pivot(index=['outcome', 'g_fn', 'inference_type', 'param'], columns='model',
                            values=['bias', 'emp_sd', 'se', 'coverage'])
        base = out/f'comparison_table_n{n}_{distribution}'
        table.to_csv(str(base)+'.csv')
        table.to_latex(str(base)+'.tex', float_format='%.4f', na_rep='-')
    print(f'Saved comparisons to {out}')


if __name__ == '__main__':
    main()
