# Reproducing the paper

Scripts for the simulation study, the NHANES analysis, the HPC jobs, the
reports and the benchmarks, with their tests. Run every command from the
repository root.

The scripts use this checkout's package in `python/neuralplsi` (each adds
`python/` to `sys.path`), which compiles `core/` on first use with `$CXX`
(default `g++`) into `NEURALPLSI_CACHE_DIR` (default `~/.cache/neuralplsi`).
The package does not need to be installed; see
[`python/README.md`](../python/README.md#running-from-a-checkout). After an
editable install of this checkout (`python -m pip install -e .`), the scripts
import that install instead, with the same Python files, and use the library it
built with CMake. The scripts
need a C++17 compiler and the packages in `requirements.txt`:

```bash
python -m pip install -r reproduce/requirements.txt
```

## Simulation study

`main_simulation.py` runs one scenario for one or more estimators. For each
replicate seed `s` it draws `2n` observations from
`neuralplsi.simulation.simulate_data` with seed `s`, fits on `n` of them and
records test-set MSE (continuous), AUC (binary) or C-index (Cox) on the other
`n`. It then evaluates `g` on a grid (1,000 points on [-3, 3]) and runs the
bootstrap with random state `s`. NeuralPLSI uses the ELU activation
(`--activation`) and fit seed `--fit_seed` (0); SplinePLSI uses fit seed `s`.

`--models` takes `NeuralPLSI` (L-BFGS to convergence), `PLSI` (SplinePLSI),
`NeuralPLSI-AdamGamma` and `NeuralPLSI-AdamFrozen` (minibatch AdamW with early
stopping: at most 200 epochs, batch size 64, learning rate 1e-3, weight decay
1e-4, patience 10; then the exact gamma fit, or the frozen correction of beta,
gamma and the intercept, on all rows) or `all` (NeuralPLSI and SplinePLSI).
The scenario is set by `--n_instances`, `--g_fn` (`linear`, `sigmoid`, `threshold`),
`--outcome` (`continuous`, `binary`, `cox`) and `--exposure_dist` (`normal`,
`uniform`, `t`). The defaults are 500 replicates (`--n_replicates`) and B = 100
(`--n_bootstrap`). The current results in `results/simulation` use n = 2000 and
500 replicates, with B = 1000 for the three NeuralPLSI estimators and B = 999
for SplinePLSI, whose earlier run was not repeated. `--bootstrap_jobs` sets the
bootstrap threads (`auto` by default).

A small local run, which takes a few seconds:

```bash
python reproduce/main_simulation.py --models all --n_instances 200 \
  --n_replicates 2 --n_bootstrap 4 --bootstrap_jobs 2 --out output/simulation-check
```

Each estimator and scenario has one result file,
`<out>/simulation+<model>+<n>+<g_fn>+<outcome>+<exposure_dist>+init0.json`
(`init1` with `--initial`), and `--out` defaults to `output/simulation`. Every
replicate records its run configuration: the settings, the software versions
and a SHA-256 of the simulation, package and engine sources. Completed
replicates are saved atomically (`--save_every`, default 1). A failed replicate
stops the run with its seed; earlier replicates are kept. A lock file rejects a
second writer to the same result file. A run resumes into an existing file only
if every recorded configuration matches, so use a fresh `--out` after changing
code or inference settings, and never combine old and new results.

Other options: `--save_bootstrap_samples` keeps the beta and gamma draws of each
replicate in compressed NPZ files under `<out>/bootstrap_samples`.
`--nn_bootstrap_quantile weibull` changes NeuralPLSI's percentile
interpolation for a sensitivity run; it changes the endpoints, not the bias of
the estimator, and is not a coverage repair. `--initial` starts NeuralPLSI
from a linear-model fit. `run_ablation_grid.py` runs NeuralPLSI with and
without `--initial`, and SplinePLSI, for n = 500 and 2000 and all nine
scenarios (`--dry-run` prints the 54 commands).

### HPC (Slurm)

`run_simulation.slurm` (n = 1000) and `run_simulation_2000.slurm` (n = 2000) are
array jobs that call `run_simulation_hpc.sh <n>`. Array IDs 0-8 run NeuralPLSI,
9-17 SplinePLSI, 18-26 `NeuralPLSI-AdamFrozen` and 27-35
`NeuralPLSI-AdamGamma`. In each block
of nine, the first three IDs use the sigmoid g, the next three the threshold g and the
last three linear g, each with continuous, binary and cox outcomes in that
order (ID 13 is SplinePLSI, threshold g, binary). The scripts default to IDs 0-8.
Submit from the repository root:

```bash
mkdir -p logs
export N_BOOTSTRAP=1000 N_REPLICATES=500 OUT_DIR=output/simulation
sbatch reproduce/run_simulation_2000.slurm                 # NeuralPLSI
sbatch --array=9-17 reproduce/run_simulation_2000.slurm    # SplinePLSI
sbatch --array=18-26 reproduce/run_simulation_2000.slurm   # NeuralPLSI-AdamFrozen
sbatch --array=27-35 reproduce/run_simulation_2000.slurm   # NeuralPLSI-AdamGamma
```

| Variable | Default | Meaning |
|---|---|---|
| `N_BOOTSTRAP` | 100 | Bootstrap draws per replicate. |
| `N_REPLICATES`, `SEED0` | 500, 0 | Replicates and the first seed. |
| `OUT_DIR` | `output/simulation` | Result directory. |
| `BOOTSTRAP_JOBS` | `SLURM_CPUS_PER_TASK`, else 1 | Native bootstrap threads, for both estimators. Lower it when memory is limited. |
| `SAVE_BOOTSTRAP_SAMPLES` | 0 | 1 keeps the beta and gamma draws in NPZ files. |
| `NPLSI_INITIAL` | 0 | 1 adds `--initial`. |
| `NN_BACKEND` | `cpp` | The only backend. |
| `NPLSI_PROJECT_DIR` | the submission directory | The checkout to run. |
| `NPLSI_PYTHON` | `python`; `/ext3/miniconda3/bin/python` in a container | The interpreter. |
| `NPLSI_CONTAINER`, `NPLSI_OVERLAY` | unset | A Singularity image and an overlay mounted read-only. |
| `NPLSI_DRY_RUN` | 0 | 1 prints the command instead of running it, including the `singularity exec` prefix when `NPLSI_CONTAINER` is set. |

The jobs limit BLAS and OpenMP to one thread. Without `NPLSI_CONTAINER` they use
the active Python environment. To use the Singularity environment on
BigPurple, set before submitting:

```bash
export NPLSI_PROJECT_DIR=/gpfs/scratch/doh03/NeuralPLSI
export NPLSI_CONTAINER=/gpfs/scratch/doh03/containers/nvidia-cuda12.9.sif
export NPLSI_OVERLAY=/gpfs/scratch/doh03/containers/survmix.ext3
export NPLSI_PYTHON=/ext3/miniconda3/bin/python
```

The C++ compiler must be available inside the container. The first job compiles
the engine into the cache; a file lock makes concurrent jobs wait for it. The
`#SBATCH` resource lines (20 CPUs, 32 GB, 12 hours, partition `cpu_short`,
excluded nodes) are for BigPurple and may need changes elsewhere; options on
the `sbatch` command line override them. The current results were run with
`--cpus-per-task=40` added to each `sbatch` line, so the bootstrap used 40
threads, and SplinePLSI with `N_BOOTSTRAP=999`. Every bootstrap draw is a
complete fit, so check the wall-time limit before a full run. A larger B
reduces bootstrap Monte Carlo error; it does not guarantee nominal coverage.

To see the command an array ID runs, without Slurm:

```bash
NPLSI_DRY_RUN=1 SLURM_ARRAY_TASK_ID=13 N_BOOTSTRAP=1000 bash reproduce/run_simulation_hpc.sh 2000
```

### Reports

```bash
python reproduce/plot_inference.py --input output/simulation --out results/simulation
python reproduce/visualize_simulation.py --input output/simulation --out results/simulation --no-plots
python reproduce/table_simulation_comparison.py --input output/simulation --out results/simulation
```

- `plot_inference.py` draws one figure, `inference.png` and `.pdf`, that
  compares the estimators: forest plots of the bias (in empirical SDs) and the interval coverage, each with its
  Monte-Carlo interval, then SE calibration for each coordinate of beta and gamma, and the estimated g with a
  ±1 SD band against the true g. `inference.csv` has the numbers, and the RMSE of each coordinate and of g. Without `--out` it writes to
  `<input>/figures`.
- `visualize_simulation.py` writes `summary_table.csv` and `.tex`,
  `summary_aggregated.csv` and per-setting tables in `tables/`, and, without
  `--no-plots`, curve plots of g.
- `table_simulation_comparison.py` writes
  `comparison_table_n<n>_<exposure_dist>.csv` and `.tex`, with the estimators
  side by side, and `comparison_metrics.csv`.
- Without `--out`, these two write into the `--input` directory.

The reports keep sample sizes and exposure distributions apart, compute
coverage from the saved interval endpoints, and give per-coefficient bias,
empirical SD, mean bootstrap SE, bootstrap centering bias and Monte Carlo SEs.
They refuse to pool result files with different run configurations. In curve
plots the shading is the spread across simulation replicates, not a bootstrap
band. The summaries of the current n = 2000 run are in
[`results/simulation`](../results/simulation/README.md).

## NHANES analysis

`NHANES/12940_2020_644_MOESM2_ESM.csv` is the NHANES dataset from the
[supplementary material](https://static-content.springer.com/esm/art%3A10.1186%2Fs12940-020-00644-4/MediaObjects/12940_2020_644_MOESM2_ESM.csv)
of Wang et al. (2020), *Environmental Health* 19:96, open access under CC BY 4.0.

```bash
python reproduce/nhanes_comparison.py
python reproduce/nhanes.py
```

`nhanes_comparison.py` fits NeuralPLSI and SplinePLSI with B = 1000 each and
writes the comparison table, the g curves and three figures to `output/nhanes`.
`nhanes.py` fits NeuralPLSI alone and writes `output/nhanes_nplsi_results.csv`
and `output/nhanes/nhanes.png`. Both read `NHANES/` and write `output/`
relative to the working directory. Each took about 1.5 minutes on a 10-core
laptop. The current results and findings are in
[`results/nhanes`](../results/nhanes/README.md).

## Benchmarks

`benchmark_backends.py` times native fits, predictions and bootstrap refits (its
name is historical; it benchmarks only the C++ engine). Engine loading is timed
separately. It writes `output/benchmark_native.json` (`--out`); `--jobs 1` is
serial:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/benchmark_backends.py --n 2000 --repeats 3 --bootstrap 24 --jobs 4
```

`compare_dense_kernels.py` compiles the pre-optimization engine from the git
history (commit `b743336`, `models/native/core.cpp`) next to the current one,
alternates their execution order, and requires byte-identical fitted states and
identical predictions, draws, SEs and intervals. It writes
`output/dense_kernel_benchmark.json` (`--out`). It works only at the
dense-kernel commit `7de7361`: from the AdamW baseline (`f95d683`) on, the model
configuration and state format differ from the baseline's, and on the current
tree it stops because the baseline has no `nplsi_version`. In a separate clone:

```bash
git clone https://github.com/hyungrok-do/NeuralPLSI.git nplsi-7de7361
cd nplsi-7de7361
git checkout 7de7361
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/compare_dense_kernels.py --repeats 3 --bootstrap 4 --jobs 1 4
```

The measured results are in
[`results/dense_kernel_benchmark.md`](../results/dense_kernel_benchmark.md),
and the earlier timing and memory history is in
[`core/README.md`](../core/README.md#timings). None of these checks estimates
coverage; that needs the full scenarios.

`native_memory_check.cpp` measures the peak RSS of a native process without
Python (arguments: n, B, workers, max_iter; B = 0 measures one fit) and prints
it as `peak_rss_mib`. On Linux it reads `VmHWM` from `/proc/self/status`, on
macOS `ru_maxrss` from `getrusage`; elsewhere it prints -1:

```bash
g++ -O3 -pthread -ffp-contract=off -std=c++17 -I core/include core/src/neuralplsi.cpp \
  core/src/spline.cpp reproduce/native_memory_check.cpp -o /tmp/nplsi-memory
/tmp/nplsi-memory 100000 4 4 5000
```

`figures/` holds the figures of an earlier initialization ablation, last
updated in commit `36dfe08`. They predate the full-batch L-BFGS estimator, and
the script that drew them has been removed.

## Changes to the study

- Simulation data keep the historical seeded draws without changing the
  caller's NumPy random state. Native initialization uses the versioned C++
  random number protocol, so fits after that change need fresh result
  directories.
- NHANES preprocessing excludes incomplete observations explicitly, and the
  reported SEs are bootstrap sample SDs. Curve grids are on the index scale, and
  intervals for the curve plus the intercept include their joint bootstrap
  uncertainty. Plots keep the actual interval endpoints even when the point
  estimate lies outside the interval.

## Tests

```bash
python -m pip install -r reproduce/requirements.txt pytest
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest reproduce/tests
```

They check that checkpoints survive serialization errors and failed
replicates, that a second writer fails, that resuming rejects mixed, legacy or
duplicate results, that reports use the recorded intervals, grids and sample
sizes, the NHANES complete-case handling, SEs and interval endpoints, and, in a
subprocess that blocks every Torch import, the native workflow and the
reproduce modules.
