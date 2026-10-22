"""One inference figure comparing NeuralPLSI and SplinePLSI simulation results.

Usage:
    python reproduce/plot_inference.py --input output/simulation

Writes <input>/figures/inference.{png,pdf}, one figure with four panels:
  A  95% bootstrap CI coverage of beta and gamma
  B  mean bootstrap SE / empirical SD
  C  estimation accuracy: RMSE of beta, gamma and g (over |u| <= 2)
  D  pointwise coverage of the bootstrap band for g

and inference.csv, the block-level numbers shown in panels A-C plus the RMSE
of the bagged estimate (the mean of the bootstrap refits).

Each estimator (and sample size, when several are present) is one series.
Monte-Carlo error for block coverage is computed across replicates: each
replicate contributes its mean coverage over the block's coordinates, because
coordinates within one fit are dependent.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce.result_io import true_g

SHAPES = ['linear', 'sfun', 'sigmoid']
OUTCOMES = ['continuous', 'binary', 'cox']
SHAPE_LABEL = {'linear': 'Linear', 'sfun': 'S-shape', 'sigmoid': 'Sigmoid'}
OUTCOME_LABEL = {'continuous': 'Continuous', 'binary': 'Binary', 'cox': 'Cox'}
MODEL_LABEL = {'NeuralPLSI': 'NeuralPLSI', 'PLSI': 'SplinePLSI'}
MODEL_COLOR = {'NeuralPLSI': '#0072B2', 'PLSI': '#D55E00'}
SIZE_MARKER = ['o', 's', '^', 'D']
SYMBOL = {'beta': r'$\beta$', 'gamma': r'$\gamma$', 'g': r'$g$'}
NOMINAL, Z = 0.95, 1.959964
DENSE = 2.  # |u| <= 2 holds most of the index distribution (SD about 1.1)
REF_LINE = dict(color='#333333', lw=1.0, ls=(0, (4, 2.5)))
MC_SHADE = dict(color='#94a3b8', alpha=0.13, lw=0, zorder=0)
SPARSE_SHADE = '#e2e8f0'

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 11,
    'axes.titlesize': 12.5, 'axes.labelsize': 11.5,
    'xtick.labelsize': 10, 'ytick.labelsize': 10.5, 'legend.fontsize': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#444444',
    'xtick.color': '#444444', 'ytick.color': '#444444',
    'figure.dpi': 110, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def load_cells(directory):
    cells = {}
    for path in sorted(Path(directory).glob('simulation+*.json')):
        _, model, n, shape, outcome, *_ = path.name.split('+')
        rows = json.loads(path.read_text())
        if model not in MODEL_LABEL or not rows:
            continue
        arr = lambda k: np.array([r[k] for r in rows], dtype=float)
        cell = {k: arr(k) for k in ['beta_est', 'beta_se', 'beta_cov', 'beta_bias',
                                    'gamma_est', 'gamma_se', 'gamma_cov', 'gamma_bias',
                                    'beta_bootstrap_mean', 'gamma_bootstrap_mean',
                                    'g_lb', 'g_ub', 'g_pred']}
        cell['grid'] = np.asarray(rows[0]['g_grid'], dtype=float)
        truth = true_g(cell['grid'], shape)
        if outcome == 'cox':
            truth = truth-float(true_g(0., shape))  # both estimators anchor g(0) = 0
        cell['g_truth'] = truth
        cells[(model, int(n), shape, outcome)] = cell
    if not cells:
        raise SystemExit(f'No NeuralPLSI or SplinePLSI result files in {directory}')
    return cells


def block_summary(cell, par):
    if par == 'g':
        dense = np.abs(cell['grid']) <= DENSE
        error = cell['g_pred'][:, dense]-cell['g_truth'][dense]
        return dict(rmse=np.sqrt(np.mean(error**2)))
    c = cell[f'{par}_cov'].mean(axis=1)
    truth = cell[f'{par}_est']-cell[f'{par}_bias']
    sd = cell[f'{par}_est'].std(axis=0, ddof=1)
    return dict(cov=c.mean(), mc=Z*c.std(ddof=1)/np.sqrt(len(c)),
                ratio=(cell[f'{par}_se'].mean(axis=0)/sd).mean(),
                rmse=np.sqrt(np.mean(cell[f'{par}_bias']**2)),
                rmse_bagged=np.sqrt(np.mean((cell[f'{par}_bootstrap_mean']-truth)**2)))


def write_table(cells, out):
    path = out/'inference.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'n', 'g_fn', 'outcome', 'block', 'coverage', 'coverage_mc',
                         'se_sd_ratio', 'rmse', 'rmse_bagged', 'n_reps'])
        for (model, n, shape, outcome), cell in sorted(cells.items()):
            for par in ('beta', 'gamma', 'g'):
                summary = block_summary(cell, par)
                writer.writerow([MODEL_LABEL[model], n, shape, outcome, par]
                                + [f'{summary[k]:.6g}' if k in summary else ''
                                   for k in ('cov', 'mc', 'ratio', 'rmse', 'rmse_bagged')]
                                + [len(cell['beta_est'])])
    print(f'wrote {path}')


def series_label(series, sizes):
    model, n = series
    return MODEL_LABEL[model]+(f', n = {n}' if len(sizes) > 1 else '')


def replicate_label(cells):
    counts = sorted({len(c['beta_est']) for c in cells.values()})
    return f'{counts[0]} replicates' if len(counts) == 1 else f'{counts[0]}-{counts[-1]} replicates'


def mc_band(reps):
    half = Z*np.sqrt(NOMINAL*(1-NOMINAL)/reps)
    return NOMINAL-half, NOMINAL+half


def style(series, sizes):
    model, n = series
    return dict(color=MODEL_COLOR[model], marker=SIZE_MARKER[sizes.index(n) % len(SIZE_MARKER)])


def legend_handles(series, sizes, *extras):
    handles = [Line2D([0], [0], lw=1.5, mfc='white', mew=1.6, ms=5.5,
                      label=series_label(s, sizes), **style(s, sizes)) for s in series]
    available = {
        'target': Line2D([0], [0], **REF_LINE, label='target (0.95 coverage, 1.0 SE ratio)'),
        'mc': Patch(color='#94a3b8', alpha=0.3, label='Monte-Carlo range at exact nominal'),
        'sparse': Patch(color=SPARSE_SHADE, label=f'|u| > {DENSE:g} (sparse data)'),
    }
    return handles+[available[e] for e in extras]


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{ext}', bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'wrote {out/name}.{{png,pdf}}')


def draw_forest(container, cells, series, sizes, key, xlabel, parameters=('beta', 'gamma'),
                ref=None, xlim=None, band=None, log=False):
    rows = [(par, s) for par in parameters for s in SHAPES]
    yy = np.arange(len(rows))
    offsets = dict(zip(series, np.linspace(-0.2, 0.2, len(series)) if len(series) > 1 else [0.]))
    axes = container.subplots(1, 3, sharey=True, sharex=True)
    shown = []
    for ax, outcome in zip(axes, OUTCOMES):
        if band is not None:
            ax.axvspan(*band, **MC_SHADE)
        if ref is not None:
            ax.axvline(ref, zorder=1, **REF_LINE)
        for k in range(1, len(parameters)):
            ax.axhline(k*len(SHAPES)-0.5, color='#cbd5e1', lw=0.9, zorder=1)
        for s in series:
            v, e = [], []
            for par, shape in rows:
                cell = cells.get(s+(shape, outcome))
                summary = block_summary(cell, par) if cell else None
                v.append(summary[key] if summary else np.nan)
                e.append(summary['mc'] if summary and key == 'cov' else 0.)
            shown += [x-d for x, d in zip(v, e)]+[x+d for x, d in zip(v, e)]
            ax.errorbar(v, yy+offsets[s], xerr=e, fmt='none', ecolor=MODEL_COLOR[s[0]],
                        elinewidth=1.5, capsize=2.5, zorder=2)
            ax.plot(v, yy+offsets[s], ls='', ms=5.5, mfc='white', mew=1.6, zorder=3, **style(s, sizes))
        if log:
            ax.set_xscale('log')
        ax.grid(axis='x', which='both' if log else 'major', alpha=0.16, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(OUTCOME_LABEL[outcome], pad=6)
        ax.set_xlabel(xlabel)
    if xlim is not None:  # widened to keep every estimate visible
        low, high = np.nanmin(shown), np.nanmax(shown)
        pad = .03*(max(high, xlim[1])-min(low, xlim[0]))
        axes[0].set_xlim(min(xlim[0], low-pad), max(xlim[1], high+pad))
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([SHAPE_LABEL[s] for _, s in rows])
    axes[0].invert_yaxis()
    # y in data coordinates so the labels follow invert_yaxis().
    for i, par in enumerate(parameters):
        axes[0].text(-0.30, i*len(SHAPES)+(len(SHAPES)-1)/2, SYMBOL[par], fontsize=15,
                     transform=axes[0].get_yaxis_transform(), ha='center', va='center',
                     clip_on=False)
    return axes


def draw_g_coverage(container, cells, series, sizes, reps):
    axes = container.subplots(3, 3, sharex=True, sharey=True)
    for i, outcome in enumerate(OUTCOMES):
        for j, shape in enumerate(SHAPES):
            ax = axes[i, j]
            ax.axhspan(*mc_band(reps), **MC_SHADE)
            ax.axhline(NOMINAL, zorder=1, **REF_LINE)
            for side in ((-3.2, -DENSE), (DENSE, 3.2)):
                ax.axvspan(*side, color=SPARSE_SHADE, alpha=0.5, lw=0, zorder=0)
            for s in series:
                cell = cells.get(s+(shape, outcome))
                if cell is None:
                    continue
                truth = cell['g_truth']
                covered = (cell['g_lb'] <= truth) & (truth <= cell['g_ub'])
                ax.plot(cell['grid'], covered.mean(axis=0), color=MODEL_COLOR[s[0]], lw=1.6, zorder=3,
                        ls='-' if sizes.index(s[1]) == 0 else '--')
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.16, lw=0.7)
            if i == 0:
                ax.set_title(SHAPE_LABEL[shape], pad=6)
            if j == 0:
                ax.set_ylabel(f'{OUTCOME_LABEL[outcome]}\npointwise coverage')
            if i == 2:
                ax.set_xlabel('index  $u = x^\\top\\beta$')
    return axes


def combined(cells, out):
    series = sorted({(model, n) for model, n, _, _ in cells}, key=lambda s: (s[1], s[0] != 'NeuralPLSI'))
    sizes = sorted({n for _, n in series})
    reps = min(len(c['beta_est']) for c in cells.values())
    fig = plt.figure(figsize=(15, 25), layout='constrained')
    panels = fig.subfigures(4, 1, height_ratios=[4.2, 4.2, 5.8, 9.0], hspace=0.03)
    titles = ['Coverage of 95% bootstrap CIs',
              'SE calibration: below 1.0 means the intervals are too narrow',
              f'Estimation accuracy: RMSE of the point estimates (g over |u| $\\leq$ {DENSE:g})',
              'Pointwise coverage of the 95% bootstrap band for g(u)']
    draw_forest(panels[0], cells, series, sizes, 'cov', 'bootstrap CI coverage',
                ref=NOMINAL, xlim=(0.85, 1.01), band=mc_band(reps))
    draw_forest(panels[1], cells, series, sizes, 'ratio', 'mean bootstrap SE / empirical SD',
                ref=1.0, xlim=(0.7, 1.3))
    draw_forest(panels[2], cells, series, sizes, 'rmse', 'RMSE (log scale)',
                parameters=('beta', 'gamma', 'g'), log=True)
    draw_g_coverage(panels[3], cells, series, sizes, reps)
    for panel, letter, title in zip(panels, 'ABCD', titles):
        panel.suptitle(rf'$\mathbf{{{letter}}}$   {title}', x=0.0, ha='left', fontsize=14)
    size_text = ', '.join(f'n = {n}' for n in sizes)
    fig.suptitle(f'NeuralPLSI and SplinePLSI simulation inference, {size_text}, '
                 f'{replicate_label(cells)} per setting', x=0.0, ha='left', fontsize=16)
    fig.legend(handles=legend_handles(series, sizes, 'target', 'mc', 'sparse'), loc='outside lower center',
               ncol=len(series)+3, frameon=False, columnspacing=2.0)
    save(fig, out, 'inference')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='output/simulation')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    out = Path(args.out or Path(args.input)/'figures')
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.input)
    combined(cells, out)
    write_table(cells, out)


if __name__ == '__main__':
    main()
