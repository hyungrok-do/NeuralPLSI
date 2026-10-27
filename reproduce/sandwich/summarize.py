"""Summaries of the sandwich Monte Carlo: SE/SD and Wald coverage per method, setting and block.

Usage: python reproduce/sandwich/summarize.py [--input output/sandwich] [--out results/sandwich]

Reads methods.jsonl (monte_carlo.py, all three outcomes: L-BFGS with the full sandwich, AdamW +
frozen correction with the frozen sandwich, AdamW + exact gamma with the full
sandwich) and spline.jsonl (SplinePLSI, as fitted and polished), and writes
sandwich_monte_carlo.csv: for beta and gamma, the mean, smallest and largest
SE / empirical SD and Wald coverage over the coordinates (NeuralPLSI: the
median SE over the datasets; SplinePLSI: the mean), the median remaining
Newton step in SE units and the median smallest Hessian eigenvalue.
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from neuralplsi.simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA

Z = 1.959964
METHODS = {'lbfgs': 'L-BFGS, full sandwich',
           'adam_frozen': 'AdamW + frozen correction, frozen sandwich',
           'adam_exact_gamma': 'AdamW + exact gamma (comparator), full sandwich'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='output/sandwich')
    parser.add_argument('--out', default='results/sandwich')
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    table = []
    rows = [json.loads(line) for line in open(Path(args.input)/'methods.jsonl')]
    for method, label in METHODS.items():
        for outcome in ('continuous', 'binary', 'cox'):
            for g in ('linear', 'sigmoid', 'threshold'):
                cell = [r[method] for r in rows if r['outcome'] == outcome and r['g'] == g]
                for block, truth in (('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)):
                    est, se = np.array([r[block] for r in cell]), np.array([r[block+'_se'] for r in cell])
                    ratio = np.median(se, axis=0)/est.std(axis=0, ddof=1)
                    cover = (np.abs(est-truth) <= Z*se).mean(axis=0)
                    table.append(dict(arm=method, fit=label, outcome=outcome, g_fn=g, block=block, datasets=len(cell),
                        se_sd=ratio.mean(), se_sd_min=ratio.min(), se_sd_max=ratio.max(),
                        coverage=cover.mean(), coverage_min=cover.min(), coverage_max=cover.max(),
                        newton_step_median=np.median([r['newton_'+block] for r in cell]),
                        min_eigenvalue_median=np.median([r['min_eigenvalue'] for r in cell])))
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
                            newton_step_median=float('nan'),
                            min_eigenvalue_median=np.median([r['min_eig'] for r in cell])))
    path = out/'sandwich_monte_carlo.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        for row in table:
            writer.writerow({k: f'{v:.6g}' if isinstance(v, float) else v for k, v in row.items()})
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
