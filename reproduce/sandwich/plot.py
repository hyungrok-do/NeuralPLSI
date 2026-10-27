"""Coverage and SE calibration of every inference method, side by side.

Usage: python reproduce/sandwich/plot.py [--sandwich output/sandwich/methods.jsonl]
           [--simulation output/simulation] [--out results/sandwich/inference_methods.png]

For beta and gamma, per setting (continuous, binary and Cox outcomes; linear,
sigmoid and threshold g; datasets 0-199 of the main simulation's design, n =
2000): coverage of the 95% interval and median SE / empirical SD, as the mean
over coordinates with the range over coordinates. The sandwich methods come
from monte_carlo.py's rows, the bootstraps from the main simulation's
(reproduce/main_simulation.py). Methods without results for a setting (the
Cox sandwich, a run not yet made) leave a gap.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from neuralplsi.simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA

Z = 1.959964
DATASETS = 200
SETTINGS = [(o, g) for o in ('continuous', 'binary', 'cox') for g in ('linear', 'sigmoid', 'threshold')]
# (label, colour, marker, group, source, key): source 'sandwich' reads monte_carlo.py's rows
# under key, 'bootstrap' the main simulation's file for model key.
METHODS = [('SplinePLSI, bootstrap', '#D55E00', '^', 0, 'bootstrap', 'PLSI'),
           ('L-BFGS, sandwich', '#0072B2', 'o', 1, 'sandwich', 'lbfgs'),
           ('L-BFGS, bootstrap', '#56B4E9', 's', 1, 'bootstrap', 'NeuralPLSI'),
           ('SGD, no correction, sandwich', '#CC79A7', 'v', 2, 'sandwich', 'adam_exact_gamma'),
           ('SGD + correction, sandwich', '#009E73', 'D', 2, 'sandwich', 'adam_frozen'),
           ('SGD + correction, bootstrap', '#004D40', 'P', 2, 'bootstrap', 'NeuralPLSI-AdamFrozen')]


def sandwich_cells(path):
    cells = {}
    if not Path(path).exists():
        return cells
    rows = [json.loads(line) for line in open(path)]
    for _, _, _, _, source, key in METHODS:
        if source != 'sandwich':
            continue
        for o, g in SETTINGS:
            cell = [r[key] for r in rows if r['outcome'] == o and r['g'] == g and r['seed'] < DATASETS]
            if len(cell) < DATASETS:
                continue
            for block, truth in (('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)):
                est, se = np.array([r[block] for r in cell]), np.array([r[block+'_se'] for r in cell])
                cells[key, o, g, block] = ((np.abs(est-truth) <= Z*se).mean(axis=0),
                                           np.median(se, axis=0)/est.std(axis=0, ddof=1))
    return cells


def bootstrap_cells(directory):
    cells = {}
    for _, _, _, _, source, key in METHODS:
        if source != 'bootstrap':
            continue
        for o, g in SETTINGS:
            path = Path(directory)/f'simulation+{key}+2000+{g}+{o}+normal+init0.json'
            if not path.exists():
                continue
            rows = [r for r in json.load(open(path)) if r['seed'] < DATASETS]
            for block, truth in (('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)):
                est, se = np.array([r[block+'_est'] for r in rows]), np.array([r[block+'_se'] for r in rows])
                lb, ub = np.array([r[block+'_lb'] for r in rows]), np.array([r[block+'_ub'] for r in rows])
                cells[key, o, g, block] = (((lb <= truth) & (truth <= ub)).mean(axis=0),
                                           np.median(se, axis=0)/est.std(axis=0, ddof=1))
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sandwich', default='output/sandwich/methods.jsonl')
    parser.add_argument('--simulation', default='output/simulation')
    parser.add_argument('--out', default='results/sandwich/inference_methods.png')
    args = parser.parse_args(argv)
    cells = {**sandwich_cells(args.sandwich), **bootstrap_cells(args.simulation)}
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 13), sharey=True, layout='constrained')
    # One row per setting; within it the methods in three groups with a gap between groups.
    step, gap = 0.1, 0.07
    positions, x = [], 0.
    for i, (_, _, _, group, _, _) in enumerate(METHODS):
        if i and group != METHODS[i-1][3]:
            x += gap
        positions.append(x); x += step
    positions = np.array(positions)-np.mean(positions)
    y = np.arange(len(SETTINGS))[::-1]*1.0
    band = Z*np.sqrt(.95*.05/DATASETS)
    for col, (block, label) in enumerate((('beta', 'β'), ('gamma', 'γ'))):
        for row, (stat, title) in enumerate(((0, 'coverage of the 95% interval'), (1, 'median SE / empirical SD'))):
            ax = axes[row, col]
            ref = .95 if stat == 0 else 1.
            if stat == 0:
                ax.axvspan(ref-band, ref+band, color='0.92', zorder=0)
            ax.axvline(ref, color='0.4', lw=0.8, ls='--', zorder=1)
            for k in range(1, len(SETTINGS)):
                ax.axhline(y[k]+0.5, color='0.85', lw=0.5, zorder=0)
            for (name, color, marker, _, _, key), dy in zip(METHODS, positions):
                ax.plot([], [], marker=marker, color=color, ls='', label=name if row == 0 and col == 0 else None)
                for k, (o, g) in enumerate(SETTINGS):
                    if (key, o, g, block) not in cells:
                        continue
                    values = cells[key, o, g, block][stat]
                    ax.hlines(y[k]-dy, values.min(), values.max(), color=color, lw=1.2, zorder=2)
                    ax.plot(values.mean(), y[k]-dy, marker=marker, color=color, ms=5.5, zorder=3)
            ax.set_title(f'{label}: {title}', loc='left')
            ax.set_xlim((0.86, 1.0) if stat == 0 else (0.75, 1.45))
    for ax in axes[:, 0]:
        ax.set_yticks(y, [f'{o}, {g}' for o, g in SETTINGS])
    fig.legend(loc='outside lower center', ncols=6, frameon=False, fontsize=8, columnspacing=1.2, handletextpad=0.3)
    fig.suptitle(f'NeuralPLSI inference, {DATASETS} datasets per setting (n = 2000): mean over coordinates, with '
                 'the range over coordinates; grey band: 95% Monte-Carlo range of a coverage of 0.95', fontsize=9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    missing = sorted({(name, o) for name, _, _, _, _, key in METHODS for o, g in SETTINGS
                      if (key, o, g, 'beta') not in cells})
    print('wrote', args.out, '| no results for:', ', '.join(f'{n} ({o})' for n, o in missing) or 'none')


if __name__ == '__main__':
    main()
