"""Inference figures for a NeuralPLSI-only native simulation run.

Usage:
    python reproduce/plot_native_inference.py --input output/simulation_native_v4_b999

Writes <input>/figures/inference.{png,pdf}, one figure with four panels:
  A  95% bootstrap CI coverage of beta and gamma, by n
  B  mean bootstrap SE / empirical SD, by n
  C  pointwise coverage of the bootstrap band for g
  D  joint-phase early-stopping diagnostics

Monte-Carlo error for block coverage is computed across replicates: each
replicate contributes its mean coverage over the block's coordinates, because
coordinates within one fit are dependent.
"""

import argparse
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
OUTCOME_COLOR = {'continuous': '#0072B2', 'binary': '#D55E00', 'cox': '#009E73'}
N_COLOR = {1000: '#6baed6', 2000: '#08519c'}
NOMINAL, Z = 0.95, 1.959964
# Joint-phase epochs are bimodal: collapsed fits stop at 7-8, the rest at 22+.
SHORT_FIT = 15
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
    for path in sorted(Path(directory).glob('simulation+NeuralPLSI+*.json')):
        _, _, n, shape, outcome, *_ = path.name.split('+')
        rows = json.loads(path.read_text())
        arr = lambda k: np.array([r[k] for r in rows], dtype=float)
        cell = {k: arr(k) for k in ['beta_est', 'beta_se', 'beta_cov', 'beta_bias',
                                    'gamma_est', 'gamma_se', 'gamma_cov', 'gamma_bias',
                                    'g_lb', 'g_ub', 'g_pred']}
        cell['grid'] = np.asarray(rows[0]['g_grid'], dtype=float)
        cell['epochs'] = np.array([r['fit_diagnostics']['joint_epochs'] for r in rows])
        cells[(int(n), shape, outcome)] = cell
    if not cells:
        raise SystemExit(f'No NeuralPLSI result files in {directory}')
    return cells


def block_summary(cell, par, keep=None):
    keep = np.ones(len(cell['epochs']), bool) if keep is None else keep
    c = cell[f'{par}_cov'][keep].mean(axis=1)
    sd = cell[f'{par}_est'][keep].std(axis=0, ddof=1)
    return dict(cov=c.mean(), mc=Z*c.std(ddof=1)/np.sqrt(len(c)),
                ratio=(cell[f'{par}_se'][keep].mean(axis=0)/sd).mean())


def replicate_label(cells):
    counts = sorted({len(c['epochs']) for c in cells.values()})
    return f'{counts[0]} replicates' if len(counts) == 1 else f'{counts[0]}-{counts[-1]} replicates'


def mc_band(reps):
    half = Z*np.sqrt(NOMINAL*(1-NOMINAL)/reps)
    return NOMINAL-half, NOMINAL+half


def legend_handles(sizes, *extras):
    handles = [Line2D([0], [0], marker='o', color=N_COLOR[n], lw=1.5, mfc='white', mew=1.6,
                      ms=5.5, label=f'n = {n}') for n in sizes]
    available = {
        'target': Line2D([0], [0], **REF_LINE, label='target (0.95 coverage, 1.0 SE ratio)'),
        'mc': Patch(color='#94a3b8', alpha=0.3, label='Monte-Carlo range at exact nominal'),
        'sparse': Patch(color=SPARSE_SHADE, label='|u| > 2 (sparse data)'),
    }
    return handles+[available[e] for e in extras]


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{ext}', bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'wrote {out/name}.{{png,pdf}}')


def draw_forest(container, cells, sizes, key, ref, xlabel, xlim, band=None):
    rows = [(par, s) for par in ('beta', 'gamma') for s in SHAPES]
    yy = np.arange(len(rows))
    offsets = dict(zip(sizes, np.linspace(-0.17, 0.17, len(sizes))))
    axes = container.subplots(1, 3, sharey=True, sharex=True)
    for ax, outcome in zip(axes, OUTCOMES):
        if band is not None:
            ax.axvspan(*band, **MC_SHADE)
        ax.axvline(ref, zorder=1, **REF_LINE)
        ax.axhline(len(SHAPES)-0.5, color='#cbd5e1', lw=0.9, zorder=1)
        for n in sizes:
            v, e = [], []
            for par, s in rows:
                cell = cells.get((n, s, outcome))
                summary = block_summary(cell, par) if cell else None
                v.append(summary[key] if summary else np.nan)
                e.append(summary['mc'] if summary and key == 'cov' else 0.)
            ax.errorbar(v, yy+offsets[n], xerr=e, fmt='o', ms=5.5, color=N_COLOR[n],
                        ecolor=N_COLOR[n], elinewidth=1.5, capsize=2.5, mfc='white',
                        mew=1.6, zorder=3)
        ax.set_xlim(*xlim)
        ax.grid(axis='x', alpha=0.16, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(OUTCOME_LABEL[outcome], pad=6)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([SHAPE_LABEL[s] for _, s in rows])
    axes[0].invert_yaxis()
    # y in data coordinates so the labels follow invert_yaxis().
    for i, symbol in enumerate((r'$\beta$', r'$\gamma$')):
        axes[0].text(-0.30, i*len(SHAPES)+(len(SHAPES)-1)/2, symbol, fontsize=15,
                     transform=axes[0].get_yaxis_transform(), ha='center', va='center',
                     clip_on=False)
    return axes


def draw_coverage(container, cells, sizes, reps):
    return draw_forest(container, cells, sizes, 'cov', NOMINAL, 'bootstrap CI coverage',
                       xlim=(0.6, 1.01), band=mc_band(reps))


def draw_se_calibration(container, cells, sizes):
    return draw_forest(container, cells, sizes, 'ratio', 1.0, 'mean bootstrap SE / empirical SD',
                       xlim=(0.3, 1.15))


def draw_g_coverage(container, cells, sizes, reps):
    axes = container.subplots(3, 3, sharex=True, sharey=True)
    for i, outcome in enumerate(OUTCOMES):
        for j, shape in enumerate(SHAPES):
            ax = axes[i, j]
            ax.axhspan(*mc_band(reps), **MC_SHADE)
            ax.axhline(NOMINAL, zorder=1, **REF_LINE)
            # The index has SD about 1.1, so |u| > 2 holds only a few percent of the data.
            for side in ((-3.2, -2.), (2., 3.2)):
                ax.axvspan(*side, color=SPARSE_SHADE, alpha=0.5, lw=0, zorder=0)
            for n in sizes:
                cell = cells.get((n, shape, outcome))
                if cell is None:
                    continue
                truth = true_g(cell['grid'], shape)
                if outcome == 'cox':
                    truth = truth-float(true_g(0., shape))
                covered = (cell['g_lb'] <= truth) & (truth <= cell['g_ub'])
                ax.plot(cell['grid'], covered.mean(axis=0), color=N_COLOR[n], lw=1.6, zorder=3)
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


def draw_early_stop(container, cells, sizes):
    axes = container.subplots(1, 3, gridspec_kw={'width_ratios': [1.1, 1, 0.9]})
    names = [(s, o) for o in OUTCOMES for s in SHAPES]
    yy = np.arange(len(names))
    offsets = dict(zip(sizes, np.linspace(-0.17, 0.17, len(sizes))))

    ax = axes[0]
    for n in sizes:
        frac = [np.mean(cells[(n, s, o)]['epochs'] < SHORT_FIT) if (n, s, o) in cells else np.nan
                for s, o in names]
        ax.barh(yy+offsets[n], frac, height=0.32, color=N_COLOR[n], label=f'n = {n}')
    ax.set_yticks(yy)
    ax.set_yticklabels([f'{SHAPE_LABEL[s]} · {OUTCOME_LABEL[o]}' for s, o in names])
    ax.invert_yaxis()
    ax.set_xlabel(f'share of fits stopping before epoch {SHORT_FIT}')
    ax.set_title('Early-stopped point fits', pad=6)
    ax.grid(axis='x', alpha=0.16, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')

    ax = axes[1]
    for outcome in OUTCOMES:
        epochs = np.concatenate([c['epochs'] for (n, s, o), c in cells.items() if o == outcome])
        error = np.concatenate([np.abs(c['beta_bias']).max(axis=1)
                                for (n, s, o), c in cells.items() if o == outcome])
        ax.scatter(epochs, error, s=7, alpha=0.35, color=OUTCOME_COLOR[outcome], lw=0,
                   label=OUTCOME_LABEL[outcome])
    ax.axvline(SHORT_FIT, **REF_LINE)
    ax.set_xlabel('joint-phase epochs')
    ax.set_ylabel(r'max$_j\,|\hat\beta_j-\beta_j|$')
    ax.set_title(r'$\beta$ error vs. training length', pad=6)
    ax.legend(frameon=False, markerscale=2.5, loc='center right')
    ax.grid(alpha=0.16, lw=0.7)

    ax = axes[2]
    affected = [(n, s, o) for (n, s, o), c in sorted(cells.items())
                if (c['epochs'] < SHORT_FIT).sum() >= 5]
    for k, key in enumerate(affected):
        cell = cells[key]
        everything = block_summary(cell, 'beta')
        kept = block_summary(cell, 'beta', cell['epochs'] >= SHORT_FIT)
        ax.plot([everything['cov'], kept['cov']], [k, k], color='#94a3b8', lw=1.2, zorder=1)
        ax.scatter(everything['cov'], k, color=N_COLOR[key[0]], s=36, zorder=3)
        ax.scatter(kept['cov'], k, facecolor='white', edgecolor=N_COLOR[key[0]], lw=1.6, s=36, zorder=3)
    ax.axvline(NOMINAL, **REF_LINE)
    ax.set_yticks(np.arange(len(affected)))
    ax.set_yticklabels([f'{SHAPE_LABEL[s]} · {OUTCOME_LABEL[o]}\n(n = {n})' for n, s, o in affected])
    ax.invert_yaxis()
    ax.set_xlabel(r'$\beta$ coverage')
    ax.set_title(r'$\beta$ coverage with and without them', pad=6)
    ax.legend(handles=[Line2D([0], [0], marker='o', ls='', color='#475569', label='all fits'),
                       Line2D([0], [0], marker='o', ls='', mfc='white', color='#475569', mew=1.6,
                              label='excluding early-stopped')],
              frameon=False, loc='lower left')
    ax.grid(axis='x', alpha=0.16, lw=0.7)
    return axes


def combined(cells, sizes, reps, out):
    fig = plt.figure(figsize=(15, 23), layout='constrained')
    panels = fig.subfigures(4, 1, height_ratios=[4.2, 4.2, 9.0, 4.6], hspace=0.03)
    titles = ['Coverage of 95% bootstrap CIs',
              'SE calibration: below 1.0 means the intervals are too narrow',
              'Pointwise coverage of the 95% bootstrap band for g(u)',
              f'Early stopping in the joint phase (fits stopping before epoch {SHORT_FIT})']
    draw_coverage(panels[0], cells, sizes, reps)
    draw_se_calibration(panels[1], cells, sizes)
    draw_g_coverage(panels[2], cells, sizes, reps)
    draw_early_stop(panels[3], cells, sizes)
    for panel, letter, title in zip(panels, 'ABCD', titles):
        panel.suptitle(rf'$\mathbf{{{letter}}}$   {title}', x=0.0, ha='left', fontsize=14)
    fig.suptitle(f'NeuralPLSI simulation inference, {replicate_label(cells)} per setting',
                 x=0.0, ha='left', fontsize=16)
    fig.legend(handles=legend_handles(sizes, 'target', 'mc', 'sparse'), loc='outside lower center',
               ncol=len(sizes)+3, frameon=False, columnspacing=2.0)
    save(fig, out, 'inference')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='output/simulation_native_v4_b999')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    out = Path(args.out or Path(args.input)/'figures')
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.input)
    sizes = sorted({n for n, _, _ in cells})
    reps = min(len(c['epochs']) for c in cells.values())
    combined(cells, sizes, reps, out)


if __name__ == '__main__':
    main()
