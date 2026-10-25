"""Median fit time and beta angle per method, outcome and n, from benchmark.py's rows.

Usage: python reproduce/scalability/summarize.py [--input output/scalability/fits.jsonl] [--out results/scalability]
"""
import argparse, collections, csv, json, statistics
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input', default='output/scalability/fits.jsonl')
    parser.add_argument('--out', default='results/scalability')
    args = parser.parse_args(argv)
    groups = collections.defaultdict(list)
    for row in map(json.loads, open(args.input)):
        groups[(row['method'], row['outcome'], row['n'])].append(row)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out/'scalability.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'outcome', 'n', 'datasets', 'time_fit_median', 'angle_median_degrees', 'time_sandwich_median'])
        for (method, outcome, n), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            sandwich = [r['time_sandwich'] for r in rows if 'time_sandwich' in r]
            writer.writerow([method, outcome, n, len(rows), f"{statistics.median(r['time_fit'] for r in rows):.4g}",
                             f"{statistics.median(r['angle'] for r in rows):.4g}",
                             f'{statistics.median(sandwich):.4g}' if sandwich else ''])
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
