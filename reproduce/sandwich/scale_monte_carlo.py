"""Coverage of AdamW + frozen correction with the frozen sandwich at large n and wide networks.

Usage: python reproduce/sandwich/scale_monte_carlo.py <n> <width> <outcome> <g> <first seed> <last seed> <out.jsonl>

Each dataset (simulate_data(n, ...), seeds first..last-1) gets an AdamW fit
with the frozen correction (the settings of monte_carlo.py, a 2-layer network of
the given width, fit seed 1337) and its frozen sandwich; for networks of at most
8,000 weights also the full sandwich of the same fit. Rows already in the output
are skipped. Runs one process per CPU (SLURM_CPUS_PER_TASK, or 4), one thread each.
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'python'), HERE]
ADAM = dict(optimizer='adam_frozen', max_iter=200, batch_size=64, learning_rate=1e-3, weight_decay=1e-4, patience=10)


def task(args):
    from neuralplsi import NeuralPLSI
    from neuralplsi.simulation import simulate_data
    n, width, outcome, g, seed = args
    X, Z, y, *_ = simulate_data(n, outcome=outcome, g_type=g, seed=seed)
    start = time.time()
    m = NeuralPLSI(family=outcome, hidden_units=width, **ADAM).fit(X, Z, y, random_state=1337)
    time_fit = time.time()-start
    start = time.time()
    frozen = m.inference_sandwich(X, Z, y, method='frozen')
    row = dict(n=n, width=width, outcome=outcome, g=g, seed=seed, params=int(m.net.config[7]),
               beta=m.beta.tolist(), gamma=m.gamma.tolist(), time_fit=time_fit, time_sandwich=time.time()-start,
               epochs=m.fit_diagnostics['iterations'], correction_iterations=m.fit_diagnostics['correction_iterations'],
               frozen=dict(beta_se=frozen['beta_se'].tolist(), gamma_se=frozen['gamma_se'].tolist(),
                           newton_beta=frozen['newton_beta'], newton_gamma=frozen['newton_gamma']))
    if row['params'] <= 8000 and (outcome != 'cox' or n*row['params'] <= 1e8):
        full = m.inference_sandwich(X, Z, y, method='full', n_jobs=1)
        row['full'] = dict(beta_se=full['beta_se'].tolist(), gamma_se=full['gamma_se'].tolist())
    return row


if __name__ == '__main__':
    n, width, outcome, g, first, last, out = sys.argv[1:8]
    n, width, first, last = int(n), int(width), int(first), int(last)
    done = {json.loads(line)['seed'] for line in open(out)} if os.path.exists(out) else set()
    jobs = [(n, width, outcome, g, s) for s in range(first, last) if s not in done]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))) as pool, open(out, 'a') as f:
        for row in pool.imap_unordered(task, jobs):
            f.write(json.dumps(row)+'\n'); f.flush()
            print(row['seed'], f"fit {row['time_fit']:.1f}s sandwich {row['time_sandwich']:.1f}s", flush=True)
