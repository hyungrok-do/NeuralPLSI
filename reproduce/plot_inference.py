"""One inference figure comparing NeuralPLSI and SplinePLSI simulation results.

Usage:
    python reproduce/plot_inference.py --input output/simulation

Writes <input>/figures/inference.{png,pdf}, one figure with four panels:
  A  95% bootstrap CI coverage of each coordinate of beta and gamma
  B  mean bootstrap SE / empirical SD of each coordinate
  C  estimation accuracy: RMSE of each coordinate, and of g over |u| <= 2
  D  pointwise coverage of the bootstrap band for g

and inference.csv, the per-coordinate numbers shown in panels A-C, block
averages (coordinate 'all') and the RMSE of the bagged estimate (the mean of
the bootstrap refits).

Each estimator (and sample size, when several are present) is one series.
The shaded band in panel A is the Monte-Carlo range of one coordinate's
coverage at exact 0.95. Monte-Carlo error for block coverage is computed
across replicates: each replicate contributes its mean coverage over the
block's coordinates, because coordinates within one fit are dependent.
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
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]  # this checkout's reproduce/ and neuralplsi
from reproduce.result_io import true_g

SHAPES = ['linear', 'sfun', 'sigmoid']
OUTCOMES = ['continuous', 'binary', 'cox']
SHAPE_LABEL = {'linear': 'Linear', 'sfun': 'S-shape', 'sigmoid': 'Sigmoid'}
OUTCOME_LABEL = {'continuous': 'Continuous', 'binary': 'Binary', 'cox': 'Cox'}
MODEL_LABEL = {'NeuralPLSI': 'NeuralPLSI', 'PLSI': 'SplinePLSI',
               'NeuralPLSI-Adam': 'NeuralPLSI, minibatch AdamW with early stopping',
               'NeuralPLSI-AdamGamma': 'the same AdamW fit, then the exact γ fit'}
MODEL_COLOR = {'NeuralPLSI': '#0072B2', 'PLSI': '#D55E00', 'NeuralPLSI-Adam': '#009E73',
               'NeuralPLSI-AdamGamma': '#CC79A7'}
MODEL_ORDER = {'NeuralPLSI': 0, 'PLSI': 1, 'NeuralPLSI-Adam': 2, 'NeuralPLSI-AdamGamma': 3}
# NeuralPLSI-AdamGamma shares the baseline's g, so its band coverage is dashed
# on top of the baseline's line.
MODEL_DASH = {'NeuralPLSI-AdamGamma': (0, (3, 2))}
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


def coordinate_summary(cell, par):
    hit = cell[f'{par}_cov']
    truth = cell[f'{par}_est'][0]-cell[f'{par}_bias'][0]  # the true coefficients are fixed
    sd = cell[f'{par}_est'].std(axis=0, ddof=1)
    return dict(truth=truth, cov=hit.mean(axis=0), mc=Z*hit.std(axis=0, ddof=1)/np.sqrt(len(hit)),
                ratio=cell[f'{par}_se'].mean(axis=0)/sd,
                rmse=np.sqrt(np.mean(cell[f'{par}_bias']**2, axis=0)),
                rmse_bagged=np.sqrt(np.mean((cell[f'{par}_bootstrap_mean']-truth)**2, axis=0)))


def write_table(cells, out):
    path = out/'inference.csv'
    keys = ('cov', 'mc', 'ratio', 'rmse', 'rmse_bagged')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'n', 'g_fn', 'outcome', 'block', 'coordinate', 'truth', 'coverage',
                         'coverage_mc', 'se_sd_ratio', 'rmse', 'rmse_bagged', 'n_reps'])
        for (model, n, shape, outcome), cell in sorted(cells.items()):
            setting = [MODEL_LABEL[model], n, shape, outcome]
            reps = len(cell['beta_est'])
            for par in ('beta', 'gamma', 'g'):
                summary = block_summary(cell, par)
                writer.writerow(setting+[par, 'all', '']
                                + [f'{summary[k]:.6g}' if k in summary else '' for k in keys]+[reps])
                if par == 'g':
                    continue
                summary = coordinate_summary(cell, par)
                for j, truth in enumerate(summary['truth']):
                    writer.writerow(setting+[par, j+1, f'{truth:.6g}']
                                    + [f'{summary[k][j]:.6g}' for k in keys]+[reps])
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
    handles = [Line2D([0], [0], lw=1.5, mfc='white', mew=1.6, ms=5.5, ls=MODEL_DASH.get(s[0], '-'),
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


def coordinate_rows(cells, with_g):
    """(block, coordinate, axis label) for each row of panels A-C."""
    summary = {par: coordinate_summary(next(iter(cells.values())), par) for par in ('beta', 'gamma')}
    rows = [(par, j, rf'$\{par}_{{{j+1}}} = {truth:.2g}$')
            for par in ('beta', 'gamma') for j, truth in enumerate(summary[par]['truth'])]
    return rows+([('g', None, SYMBOL['g'])] if with_g else [])


def setting_axes(container, xlabel):
    """One axes per (outcome, shape), grouped under outcome headings; all share x and y."""
    # The first group also holds the row labels, so it gets more width.
    groups = container.subfigures(1, 3, wspace=0.01, width_ratios=[1.2, 1, 1])
    axes = []
    for group, outcome in zip(groups, OUTCOMES):
        group.suptitle(OUTCOME_LABEL[outcome], fontsize=12.5)
        group.supxlabel(xlabel, fontsize=11)
        for shape, ax in zip(SHAPES, group.subplots(1, 3)):
            ax.set_title(SHAPE_LABEL[shape], fontsize=11, pad=4)
            axes.append((outcome, shape, ax))
    first = axes[0][2]
    for _, _, ax in axes[1:]:
        ax.sharex(first)
        ax.sharey(first)
        ax.tick_params(labelleft=False)
    return axes


def draw_coordinates(container, cells, series, sizes, key, xlabel, with_g=False,
                     ref=None, xlim=None, band=None, log=False, step=None):
    rows = coordinate_rows(cells, with_g)
    yy = np.arange(len(rows))
    offsets = dict(zip(series, np.linspace(-0.22, 0.22, len(series)) if len(series) > 1 else [0.]))
    axes = setting_axes(container, xlabel)
    shown = []
    for outcome, shape, ax in axes:
        if band is not None:
            ax.axvspan(*band, **MC_SHADE)
        if ref is not None:
            ax.axvline(ref, zorder=1, **REF_LINE)
        for k in range(1, len(rows)):
            if rows[k][0] != rows[k-1][0]:
                ax.axhline(k-0.5, color='#cbd5e1', lw=0.9, zorder=1)
        for s in series:
            cell = cells.get(s+(shape, outcome))
            if cell is None:
                continue
            summary = {par: coordinate_summary(cell, par) for par in ('beta', 'gamma')}
            v = [block_summary(cell, 'g')['rmse'] if par == 'g' else summary[par][key][j]
                 for par, j, _ in rows]
            shown += v
            ax.plot(v, yy+offsets[s], ls='', ms=4.5, mfc='white', mew=1.4, zorder=3, **style(s, sizes))
        ax.grid(axis='x', which='both' if log else 'major', alpha=0.16, lw=0.7)
        ax.grid(axis='y', alpha=0.08, lw=0.7)
        ax.set_axisbelow(True)
    first = axes[0][2]  # scale, ticks and limits carry over to the shared axes
    if log:  # limits first: settings without results leave empty axes
        positive = [v for v in shown if v > 0]
        first.set_xlim(min(positive)/1.5, max(positive)*1.5)
        first.set_xscale('log')
    elif step is not None:
        first.xaxis.set_major_locator(MultipleLocator(step))
    if xlim is not None:  # widened to keep every estimate visible
        low, high = np.nanmin(shown), np.nanmax(shown)
        pad = .03*(max(high, xlim[1])-min(low, xlim[0]))
        first.set_xlim(min(xlim[0], low-pad), max(xlim[1], high+pad))
    first.set_yticks(yy)
    first.set_yticklabels([label for _, _, label in rows])
    first.invert_yaxis()
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
                        ls=MODEL_DASH.get(s[0], '-') if sizes.index(s[1]) == 0 else '--')
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
    series = sorted({(model, n) for model, n, _, _ in cells}, key=lambda s: (s[1], MODEL_ORDER[s[0]]))
    sizes = sorted({n for _, n in series})
    reps = min(len(c['beta_est']) for c in cells.values())
    fig = plt.figure(figsize=(15, 29), layout='constrained')
    panels = fig.subfigures(4, 1, height_ratios=[5.6, 5.6, 6.0, 9.0], hspace=0.03)
    titles = ['Coverage of the 95% bootstrap CI for each coordinate of $\\beta$ and $\\gamma$',
              'SE calibration for each coordinate: below 1.0 means the intervals are too narrow',
              f'Estimation accuracy: RMSE of each coordinate, and of g over |u| $\\leq$ {DENSE:g}',
              'Pointwise coverage of the 95% bootstrap band for g(u)']
    draw_coordinates(panels[0], cells, series, sizes, 'cov', 'bootstrap CI coverage',
                     ref=NOMINAL, xlim=(0.88, 1.0), band=mc_band(reps), step=0.05)
    draw_coordinates(panels[1], cells, series, sizes, 'ratio', 'mean bootstrap SE / empirical SD',
                     ref=1.0, xlim=(0.8, 1.2), step=0.1)
    draw_coordinates(panels[2], cells, series, sizes, 'rmse', 'RMSE (log scale)', with_g=True, log=True)
    draw_g_coverage(panels[3], cells, series, sizes, reps)
    for panel, letter, title in zip(panels, 'ABCD', titles):
        panel.suptitle(rf'$\mathbf{{{letter}}}$   {title}', x=0.0, ha='left', fontsize=14)
    size_text = ', '.join(f'n = {n}' for n in sizes)
    fig.suptitle(f'Bootstrap inference in the simulations, {size_text}, '
                 f'{replicate_label(cells)} per setting', x=0.0, ha='left', fontsize=16)
    handles = legend_handles(series, sizes, 'target', 'mc', 'sparse')
    rows = 1 if len(series) < 3 else 2
    ncol = -(-len(handles)//rows)
    # Legends fill column by column; reorder so each row reads left to right.
    handles = [handles[r*ncol+c] for c in range(ncol) for r in range(rows) if r*ncol+c < len(handles)]
    fig.legend(handles=handles, loc='outside lower center', ncol=ncol, frameon=False, columnspacing=2.0)
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
