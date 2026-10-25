"""Summaries of the sandwich Monte Carlo: SE/SD and Wald coverage per arm, setting and block.

Usage: python reproduce/sandwich/summarize.py [--input output/sandwich] [--out results/sandwich]

Reads <arm>.jsonl for the arms converged, current, adam and adamgamma, and
spline.jsonl (SplinePLSI, as fitted and polished), and writes
sandwich_monte_carlo.csv: for beta and gamma, the mean, smallest and largest
sandwich SE / empirical SD and Wald coverage over the coordinates, and the fits'
median gradient norm and smallest Hessian eigenvalue.
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from neuralplsi.simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA

Z = 1.959964
ARMS = {'converged': 'L-BFGS, converged (1e-9 over 20 iterations)', 'current': 'L-BFGS, package stopping rule (1e-5 over 10)',
        'adam': 'minibatch AdamW with early stopping', 'adamgamma': 'AdamW, then the exact gamma fit'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='output/sandwich')
    parser.add_argument('--out', default='results/sandwich')
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    table = []
    for arm, label in ARMS.items():
        path = Path(args.input)/f'{arm}.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in open(path)]
        for outcome in ('continuous', 'binary'):
            for g in ('linear', 'sigmoid', 'threshold'):
                cell = [r for r in rows if r['outcome'] == outcome and r['g'] == g]
                for block, truth in (('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)):
                    est, se = np.array([r[block] for r in cell]), np.array([r['se_'+block] for r in cell])
                    ratio = se.mean(axis=0)/est.std(axis=0, ddof=1)
                    cover = (np.abs(est-truth) <= Z*se).mean(axis=0)
                    table.append(dict(arm=arm, fit=label, outcome=outcome, g_fn=g, block=block, datasets=len(cell),
                        se_sd=ratio.mean(), se_sd_min=ratio.min(), se_sd_max=ratio.max(),
                        coverage=cover.mean(), coverage_min=cover.min(), coverage_max=cover.max(),
                        grad_norm_median=np.median([r['grad_norm'] for r in cell]),
                        min_eigenvalue_median=np.median([r['min_eig'] for r in cell]),
                        hessian_not_pd=np.mean([r['min_eig'] <= 0 for r in cell]),
                        time_fit_median=np.median([r['time_fit'] for r in cell]),
                        time_sandwich_median=np.median([r['time_sandwich'] for r in cell])))
    # SplinePLSI: knots at their fitted values; as fitted and after Newton polishing.
    path = Path(args.input)/'spline.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in open(path)]
        for prefix, arm, label in (('', 'spline', 'SplinePLSI, as fitted'), ('polished_', 'spline_polished', 'SplinePLSI, polished')):
            for outcome in ('continuous', 'binary'):
                for g in ('linear', 'sigmoid', 'threshold'):
                    cell = [r for r in rows if r['outcome'] == outcome and r['g'] == g]
                    for block, truth in (('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)):
                        est = np.array([r[prefix+block] for r in cell])
                        se = np.array([r[prefix+'se_'+block] for r in cell])
                        ratio = se.mean(axis=0)/est.std(axis=0, ddof=1)
                        cover = (np.abs(est-truth) <= Z*se).mean(axis=0)
                        table.append(dict(arm=arm, fit=label, outcome=outcome, g_fn=g, block=block, datasets=len(cell),
                            se_sd=ratio.mean(), se_sd_min=ratio.min(), se_sd_max=ratio.max(),
                            coverage=cover.mean(), coverage_min=cover.min(), coverage_max=cover.max(),
                            grad_norm_median=np.median([r[prefix+'grad_norm'] for r in cell]),
                            min_eigenvalue_median=np.median([r['min_eig'] for r in cell]),
                            hessian_not_pd=np.mean([r['min_eig'] <= 0 for r in cell]),
                            time_fit_median=np.median([r['time_fit'] for r in cell]),
                            time_sandwich_median=np.median([r['time_sandwich'] for r in cell])))
    path = out/'sandwich_monte_carlo.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        for row in table:
            writer.writerow({k: f'{v:.6g}' if isinstance(v, float) else v for k, v in row.items()})
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
