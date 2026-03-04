import os
import itertools
from joblib import Parallel, delayed
import sys

# Make sure we can import from the main directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reproduce.visualize_ablation import run_and_save_simulation

# Grid specifications
n_values = [500, 2000]
outcomes = ['continuous', 'binary', 'cox']
g_fns = ['linear', 'sfun', 'sigmoid']
init_flags = [False, True]
model_types = ['NeuralPLSI', 'PLSI']
n_runs = 50

# Ensure output directory exists
os.makedirs('reproduce/output', exist_ok=True)

def task(outcome, g_fn, init, model, n, n_runs):
    # Skip invalid combinations
    if init and model == 'PLSI':
        return
        
    # The run_and_save_simulation generates a specific metric JSON and prints results. 
    # Use 1 worker per task since joblib handles overall parallelism.
    try:
        run_and_save_simulation(outcome=outcome, g_fn=g_fn, init=init, model=model, 
                                n=n, n_runs=n_runs, n_jobs=1)
    except Exception as e:
        print(f"Error in task: {outcome}, {g_fn}, {init}, {model}, {n} -> {e}")

# Build parameters
params = []
for outcome in outcomes:
    for g_fn in g_fns:
        for init in init_flags:
            for model in model_types:
                for n in n_values:
                    # Filter out PLSI + init=True
                    if init and model == 'PLSI':
                        continue
                    params.append((outcome, g_fn, init, model, n, n_runs))

print(f"Total configurations to run: {len(params)}")

# Run all combinations in parallel. Use n_jobs matching SLURM cpus-per-task.
# For HPC, using all available threads is optimal.
n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
print(f"Running on {n_cores} cores.")

Parallel(n_jobs=n_cores)(
    delayed(task)(*p) for p in params
)

print("Ablation grid completed successfully.")
