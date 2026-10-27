"""The large-n, wide-network study (scale_monte_carlo.py rows): coverage and SE/SD per cell.

Usage: python reproduce/sandwich/summarize_scale.py [--input output/sandwich/scale] [--out results/sandwich]

Writes scale_coverage.csv: per n, width, outcome, g, block and sandwich
(frozen; full where the network has at most 8,000 weights), the Wald coverage
and the median SE / empirical SD averaged over coordinates, the bias of the
estimate in units of its SD averaged over coordinates, and median times.
"""
import argparse, csv, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
from neuralplsi.simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA

Z = 1.959964


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='output/sandwich/scale')
    parser.add_argument('--out', default='results/sandwich')
    args = parser.parse_args(argv)
    table = []
    for path in glob.glob(str(Path(args.input)/'*.jsonl')):
        rows = [json.loads(line) for line in open(path)]
        if not rows:
            continue
        r0 = rows[0]
        for block, truth in (('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)):
            est = np.array([r[block] for r in rows]); sd = est.std(axis=0, ddof=1)
            for sandwich in ('frozen', 'full'):
                if sandwich not in r0:
                    continue
                se = np.array([r[sandwich][block+'_se'] for r in rows])
                table.append(dict(n=r0['n'], width=r0['width'], params=r0['params'], outcome=r0['outcome'], g_fn=r0['g'],
                                  block=block, sandwich=sandwich, datasets=len(rows),
                                  coverage=(np.abs(est-truth) <= Z*se).mean(),
                                  se_sd=(np.median(se, axis=0)/sd).mean(),
                                  bias_sd=(np.abs(est.mean(axis=0)-truth)/sd).mean(),
                                  time_fit_median=np.median([r['time_fit'] for r in rows]),
                                  epochs_median=np.median([r['epochs'] for r in rows])))
    table.sort(key=lambda r: (r['n'], r['width'], r['outcome'], r['g_fn'], r['block'], r['sandwich']))
    out = Path(args.out)/'scale_coverage.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows({k: f'{v:.6g}' if isinstance(v, float) else v for k, v in row.items()} for row in table)
    print('wrote', out)


if __name__ == '__main__':
    main()
