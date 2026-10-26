"""The matched study: sandwich and bootstrap on the same fits (study.py rows).

Usage: python reproduce/sandwich/summarize_matched.py [--input output/sandwich/matched] [--out results/sandwich]

Writes matched_comparison.csv: per arm, setting and block, the median SE over
the datasets divided by the empirical SD (averaged over coordinates) for the
sandwich and the bootstrap, the coverage of the sandwich's Wald interval and of
the bootstrap's percentile interval (over coordinates and datasets), the
median ratio of their interval widths and of the two SEs per fit, and median times.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from neuralplsi.simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA

Z = 1.959964


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='output/sandwich/matched')
    parser.add_argument('--out', default='results/sandwich')
    args = parser.parse_args(argv)
    table = []
    for arm in ('converged', 'current'):
        for outcome in ('continuous', 'binary'):
            for g in ('linear', 'sigmoid', 'threshold'):
                rows = [json.loads(line) for line in open(Path(args.input)/f'{arm}_{outcome}_{g}.jsonl')]
                for block, truth in (('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)):
                    est = np.array([r[block] for r in rows]); sd = est.std(axis=0, ddof=1)
                    sw, bs = np.array([r['sandwich_'+block] for r in rows]), np.array([r['boot_se_'+block] for r in rows])
                    lb, ub = np.array([r['boot_lb_'+block] for r in rows]), np.array([r['boot_ub_'+block] for r in rows])
                    table.append(dict(arm=arm, outcome=outcome, g_fn=g, block=block, datasets=len(rows),
                        se_sd_sandwich=(np.median(sw, axis=0)/sd).mean(), se_sd_bootstrap=(np.median(bs, axis=0)/sd).mean(),
                        coverage_sandwich=(np.abs(est-truth) <= Z*sw).mean(), coverage_bootstrap=((lb <= truth) & (truth <= ub)).mean(),
                        width_ratio=np.median(2*Z*sw)/np.median(ub-lb), se_ratio_per_fit=np.median(sw/bs),
                        time_fit=np.median([r['time_fit'] for r in rows]), time_sandwich=np.median([r['time_sandwich'] for r in rows]),
                        time_bootstrap_500_40_threads=np.median([r['time_bootstrap'] for r in rows])))
    path = Path(args.out)/'matched_comparison.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        for row in table:
            writer.writerow({k: f'{v:.6g}' if isinstance(v, float) else v for k, v in row.items()})
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
