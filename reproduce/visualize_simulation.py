#!/usr/bin/env python3
"""Report each sample size and parameter separately, using recorded CI endpoints."""
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]  # this checkout's reproduce/ and neuralplsi
from reproduce.result_io import load_results, compute_summaries, curves, true_g


def save_summary_tables(df, out_dir):
    if df.empty:
        raise ValueError('No coefficient estimates to summarize')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/'summary_table.csv', index=False)
    grouped = df.assign(block=df.param.str.split('_').str[0]).groupby(
        ['model', 'n', 'g_fn', 'outcome', 'x_dist', 'inference_type', 'block'], dropna=False)
    aggregated = grouped.agg(MAB=('bias', lambda x: np.mean(np.abs(x))),
        Emp_SD=('emp_sd', 'mean'), SE=('se', 'mean'), Coverage=('coverage', 'mean'), N_reps=('n_reps', 'first'))
    aggregated.to_csv(out_dir/'summary_aggregated.csv')
    aggregated.to_latex(out_dir/'summary_table.tex', float_format='%.4f')
    tables = out_dir/'tables'
    tables.mkdir(exist_ok=True)
    for setting, group in df.groupby(['model', 'n', 'g_fn', 'outcome', 'x_dist', 'inference_type']):
        name = 'table+'+'+'.join(map(str, setting))
        table = group.set_index('param')[['bias', 'emp_sd', 'se', 'coverage', 'bias_mcse', 'coverage_mcse', 'bootstrap_bias', 'n_reps', 'n_intervals']]
        table.to_csv(tables/(name+'.csv'))
        table.to_latex(tables/(name+'.tex'), float_format='%.4f')


def plot_g_panels(records, out_dir):
    scenarios = sorted({(r['_outcome'], r['_x_dist'], r['_n']) for r in records})
    for outcome, distribution, n in scenarios:
        selected = [r for r in records if (r['_outcome'], r['_x_dist'], r['_n']) == (outcome, distribution, n)]
        links = sorted({r['_g_fn'] for r in selected})
        fig, axes = plt.subplots(1, len(links), figsize=(5*len(links), 4), squeeze=False)
        for ax, link in zip(axes[0], links):
            for rec in (r for r in selected if r['_g_fn'] == link):
                grid, values = curves(rec)
                if grid is None:
                    continue
                color = 'steelblue' if rec['_model'] == 'NeuralPLSI' else 'coral'
                lower, upper = np.percentile(values, [2.5, 97.5], axis=0)
                ax.fill_between(grid, lower, upper, color=color, alpha=.2)
                ax.plot(grid, values.mean(axis=0), color=color, label=rec['_model'])
                truth = true_g(grid, link)
                if outcome == 'cox':
                    truth = truth-float(true_g(0., link))
                ax.plot(grid, truth, 'k--', label='Truth' if len(ax.lines) == 1 else '_nolegend_')
            ax.set(title=link, xlabel='Index', ylabel='g(index)')
            ax.legend()
        fig.suptitle(f'{outcome}, n={n}, X~{distribution}\nShading: central 95% of replicate estimates')
        fig.tight_layout()
        fig.savefig(Path(out_dir)/f'gplot_panels_{outcome}_{distribution}_n{n}.png', dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/simulation')
    parser.add_argument('--out', default=None)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    records = load_results(args.input)
    out = Path(args.out or args.input)
    save_summary_tables(compute_summaries(records), out)
    if not args.no_plots:
        plot_g_panels(records, out)
    print(f'Summarized {len(records)} scenario files in {out}')


if __name__ == '__main__':
    main()
