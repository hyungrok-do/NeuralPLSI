# NeuralPLSI

Neural Partial-Linear Single Index (NeuralPLSI) is a method for modeling outcomes that depend on a linear combination of exposures (index) transformed by a flexible nonlinear function, adjusting for covariates.

Repository: [https://github.com/hyungrok-do/neuralplsi](https://github.com/hyungrok-do/neuralplsi)

Arxiv: [https://arxiv.org/abs/2512.11593](https://arxiv.org/abs/2512.11593)

## Overview

This package implements two models:
- **NeuralPLSI**: Uses a standalone C++ neural network to learn the nonlinear link function $g(\beta^T X)$.
- **PLSI**: Uses B-splines to estimate the link function (Wang et al., 2020).

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hyungrok-do/neuralplsi.git
cd neuralplsi
```

### 2. Install Dependencies

Use Python 3.10+ and a C++17 compiler (`g++` or `clang++`). Torch is not required.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Tutorial
See `tutorial.ipynb` for a step-by-step guide on generating data, fitting models, and visualizing results.

### 2. Simulation
The package includes a simulation suite to evaluate model performance.

```bash
# Run a simulation (e.g., n=500, sigmoid relationship, continuous outcome)
python reproduce/main_simulation.py --n_instances 500 --g_fn sigmoid --outcome continuous --models all

# Submit the scenario arrays from the repository root (see HPC settings below)
mkdir -p logs
sbatch reproduce/run_simulation.slurm
sbatch reproduce/run_simulation_2000.slurm
```

To visualize results:
```bash
python reproduce/visualize_simulation.py
```
This generates summary tables and plots in the `output/` directory.

### 3. NHANES Analysis
The NHANES dataset used in this example was downloaded from the [supplementary material](https://static-content.springer.com/esm/art%3A10.1186%2Fs12940-020-00644-4/MediaObjects/12940_2020_644_MOESM2_ESM.csv) of Wang et al. (2020).

To run the analysis (requires `NHANES/` directory):

```bash
python reproduce/nhanes.py
```
This script performs data preprocessing, model fitting, and bootstrap inference, saving results to `output/`.

## Key Components

- **`models/`**: Core implementation of `NeuralPLSI` and `SplinePLSI`.
- **`simulation/`**: Utilities for generating synthetic data.

## References

- Wang, Y., Wu, Y., Jacobson, M.H. et al. A family of partial-linear single-index models for analyzing complex environmental exposures with continuous, categorical, time-to-event, and longitudinal health outcomes. *Environ Health* **19**, 96 (2020). https://doi.org/10.1186/s12940-020-00644-4

## Cite As

```
@article{
  do2025neural,
  title={Neural Network-based Partial-Linear Single-Index Models for Environmental Mixtures Analysis},
  author={Do, Hyungrok and Wang, Yuyan and Liu, Mengling and Lee, Myeonggyun},
  journal={arXiv preprint arXiv:2512.11593},
  year={2025}
}
```

## Standalone C++ NeuralPLSI

NeuralPLSI is implemented entirely in C++17. Initialization, the network, analytic
backpropagation, AdamW, data splitting, minibatch shuffling, validation checkpoint selection,
prediction, model serialization and the entire bootstrap workflow all run in
C++. There is **no Torch or
LibTorch dependency**, including in the Python examples and tests. The Python API
passes NumPy arrays to a single native bootstrap call and formats the returned
results.

The SplinePLSI comparator is also implemented in C++ (`models/native/spline.cpp`),
in double precision. For each unit index beta, five knots sit at equally spaced
order statistics of `x'beta` (Wang et al., 2020), clamped cubic B-splines give
seven basis functions, and gamma and the spline coefficients are the exact
ridge-penalized fit on `[Z, B(x'beta)]` (alpha = 1e-6): least squares by
Cholesky, logistic regression and the Breslow Cox partial likelihood by Newton.
beta is profiled over the unit sphere by BFGS with central finite differences
(step 1e-3, which averages over the small kinks that order-statistic knots
create), followed by a compass search on the profile loss, from a seeded random
direction; `beta[0] >= 0` identifies its sign. A Cox likelihood cannot identify
the level of g, so the spline is shifted to `g(0) = 0`, as in NeuralPLSI. Its
bootstrap uses native threads and the same resampled rows as the NeuralPLSI
bootstrap for the same seed, so the two estimators are compared on identical
bootstrap samples.

At the same beta the C++ profile reproduces the former NumPy/SciPy implementation
(continuous: to machine precision; binary and Cox: equal or lower loss, because
Newton solves the inner fit exactly). On 20 simulated n = 2,000 datasets per
outcome, the full fits reached the same beta accuracy as the SciPy version for
continuous and binary outcomes, and fitted 30-100 times faster for binary and Cox
outcomes. The SciPy version's SLSQP search stopped early on Cox outcomes (mean
max |beta error| 0.31 against 0.033), which also inflated its bootstrap SEs.

```python
from models import NeuralPLSI

model = NeuralPLSI(family="binary", activation="ELU")
model.fit(X, Z, y, random_state=0)
boot = model.inference_bootstrap(X, Z, y, n_samples=999, n_jobs=8)
probability = model.predict_proba(X, Z)

model.save("exposure_model.nplsi")
restored = NeuralPLSI.load("exposure_model.nplsi")
```

Native model files retain the weights, settings and index/link orientation with
exact float32 round trips. Bootstrap intervals and draws are separate results;
they are not included in these prediction-model files.

The Python binding supports Linux/macOS with a C++17 compiler. `CXX` selects the
compiler (default `g++`). The shared library is built once, cached by source,
compiler and platform, and loaded through `ctypes`. A file lock coordinates
compilation across independent jobs; bootstrap threads share the loaded library. No Eigen, pybind11 or external numerical library is needed
by the C++ model. To prebuild inside your HPC environment:

```bash
python -m models.native
```

Use `NEURALPLSI_CACHE_DIR` to select a writable cache directory. The implementation
supports continuous, binary and Cox outcomes, ELU, LeakyReLU, ReLU and Tanh, and
optional intercepts for continuous/binary outcomes. `n_hidden_layers` and
`hidden_units` configure the network depth and common hidden width; backpropagation
loops over those layers and is not restricted to the default two layers of 32 units.
It uses float32 network
parameters and double precision for Cox risk-set accumulation and the optional
linear initialization. `backend="cpp"` and `device="cpu"` are the defaults and
only supported execution modes (`device="auto"` is a CPU alias).

### Optimization

The default `optimizer="alternating"` keeps the two hidden layers of 32 units
and fits on **all supplied rows**. Each cycle runs `link_epochs` ordinary neural
training epochs (default 1), then freezes the link for a full-data coefficient
update. `max_epoch=200` is the total neural epoch budget, including a final
partial block if needed. There is no patience-based early stopping.

| Component | Alternating update | Regularization |
|---|---|---|
| Link network and linear skip | Minibatch AdamW, base rate 0.001 | Decay 0.0001, identifying `g(0)^2` anchor |
| Index direction, beta | Full-data damped Gauss-Newton, sphere retraction and line search | No weight decay |
| Gamma and optional intercept | Joint coefficient step, then conditional Newton refinement | Existing numerical ridge on gamma, no intercept penalty |

Adam's moment coefficients remain 0.9 and 0.999 with epsilon 1e-8. Its gradient
norm limit (default 1.0) applies only to the link parameters. Beta and gamma do
not receive minibatch Adam updates or participate in that clipping calculation.
`weight_decay_beta` is used only by the comparison `joint` optimizer.

The coefficient Jacobian uses the link input derivative and projects exposure
covariates onto the tangent space of the current beta direction. The small
coefficient system includes gamma cross-curvature, uses double precision, and
adds numerical damping. A proposed direction is capped at tangent length 0.25
(an angular change of at most `atan(0.25)`), retracted onto the sphere, and checked
against the actual full-data objective by Armijo backtracking. Failed steps
are rejected and counted. Damping starts at 0.001 times the mean curvature
(with a numerical floor), decreases after acceptance, and increases after a
failed line search. The frozen network is identical for every score, curvature,
and line-search evaluation within a coefficient update.

The network objectives remain mean squared error, binary cross-entropy, or the
negative Breslow Cox partial likelihood divided by the event count, plus the
soft `g(0)^2` anchor. **Cox network minibatches still use within-batch risk sets**.
Cox coefficient updates and the conditional gamma fit use complete risk sets
for all fitting rows, including ties. Consequently this Cox algorithm is not
exact block descent on a single full partial likelihood. Stabilizing beta does
not itself remove the network's Cox approximation, statistical overfitting, or
link-estimation bias.

No holdout is used by default, and the final completed cycle is returned.
Regularization and the training budget must therefore be fixed as part of the
estimator. To retain checkpoint selection, set `validation_fraction=0.2`.
Training still runs its entire budget. Validation is evaluated at completed
cycle boundaries, and the entire best state (beta, gamma, and network) is
restored. Under alternating training, gamma also uses only fitting rows,
so held-out observations do not enter its Newton fit. Checkpoint selection is
not a guarantee of performance on future data.

```python
model = NeuralPLSI(activation="ELU", optimizer="alternating",
                   validation_fraction=0.0, link_epochs=1)
# Optional held-out checkpoint selection, without early termination:
selected = NeuralPLSI(activation="ELU", validation_fraction=0.2)
# Matched full-data comparison with the former joint optimizer:
joint_full = NeuralPLSI(activation="ELU", optimizer="joint", validation_fraction=0.0)
# Historical joint protocol, including its 80/20 split:
joint_legacy = NeuralPLSI(activation="ELU", optimizer="joint")
```

`optimizer="joint"` preserves minibatch AdamW on all parameters, its 50x gamma
learning rate, shared clipping, and beta decay. When a holdout is requested,
it preserves the historical final gamma fit on both partitions. Its reported
final `validation_loss` is therefore not fully held out. `checkpoint_loss`
records the selection score before that refit. Both optimizers retain the
existing L2 index normalization, linear skip and sign canonicalization.

`fit_diagnostics` records neural epochs, fitting/validation row counts,
checkpoint epoch and score, returned-model outcome losses, accepted/rejected
coefficient steps, attempted cycles, and the final tangent beta score norm.
Checkpoint scores include the anchor; reported training and validation losses
are outcome losses. `joint_epochs` remains an alias of `epochs` for existing
callers. `max_epoch=0` performs initialization only. `initial=True` fits a native
linear/logistic/Cox starting point using the fitting rows (all rows without a
holdout). It is off by default.

The scalar-input link derivative uses a chunked forward-mode pass. Native
workers reuse buffers and gather bootstrap samples directly into their fitting
partitions. Layer activation storage remains bounded by minibatch/chunk size;
the full-data coefficient design requires approximately `8*n*(p+q+intercept)`
bytes per worker. Coefficient curvature requires quadratic storage in the
number of coefficients, so this optimizer targets a low-dimensional index.
The packed bootstrap draws use `8*B*(p+q+intercept+grid_size)` shared bytes.

These are alternating conditional updates, not exact neural profile likelihood.
The optimizer, holdout, block length and all other fitting settings are repeated
inside every cold native bootstrap refit. Coverage still needs simulation.

To compare estimation error across datasets and sensitivity to optimizer seeds,
with matching full-data fits and unchanged network capacity:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/benchmark_optimizers.py --datasets 10 --starts 10 --jobs 4
```

This point-fit pilot includes SplinePLSI as a reference. It does not estimate
bootstrap coverage. Reported concurrent fit timings include CPU contention.
The eight-dataset/eight-start pilot is summarized in
[`results/optimizer_pilot`](results/optimizer_pilot/README.md).

### Native bootstrap

`inference_bootstrap(..., n_jobs=8)` uses up to eight C++ worker threads in the
calling process. There are no Python worker processes, per-replicate Python calls,
model serialization between workers, or Python RNG calls. C++ handles:

- Paired row resampling of exposures, covariates and outcomes.
- A fresh seeded initialization for every refit (cold start), so each draw
  re-samples the optimizer noise as well as the rows. Warm starts from the
  original fit were removed: refits inherited its optimizer error, and at
  n = 2,000 the bootstrap SE of beta was only 0.69-0.82 of its sampling SD for
  binary and Cox outcomes (0.95 for continuous), against 1.00-1.04 with cold starts.
- Refitting, sign canonicalization and evaluation of the requested link grid.
- Bootstrap means, sample SDs (`ddof=1`) and percentile endpoints in double precision.

The source model stays unchanged. Each replicate writes to its own output row,
so scheduling and worker count do not change the draws or summaries. If a fit
fails, all threads are joined and the call raises an error identifying the
replicate and fit seed. Failed draws are never silently discarded or retried.

`n_jobs='auto'` or `-1` uses available CPUs; `-2` leaves one available CPU unused.
Worker counts are capped by the number of draws, CPU affinity, detected Linux
cgroup quotas and `SLURM_CPUS_PER_TASK`. Each individual fit is single-threaded.
The result records `bootstrap_engine`, `bootstrap_rng` and `n_workers`.

Resampling now uses `splitmix64_bootstrap_v1`. Replicate `b` resamples with a
separate stream keyed by `random_state+b` and fits with `random_state+1337+b`.
This preserves reproducibility across thread counts and when extending B, but
changes the draws relative to the former NumPy RNG. Use a fresh simulation output
directory. The alternating optimizer changes the fitting rule; use fresh simulation outputs.

### Use without Python

The C API is declared in `models/native/core.h`. The example trains a nonlinear
model, runs serial and threaded bootstrap with SEs and intervals, checks that the
draws match exactly, and verifies a save/load round trip. It uses only the C++
standard library:

```bash
g++ -O3 -pthread -ffp-contract=off -std=c++17 -I models/native \
  models/native/core.cpp models/native/spline.cpp examples/native_neuralplsi.cpp \
  -o /tmp/neuralplsi_demo
/tmp/neuralplsi_demo /tmp/example.nplsi
```

An optional CMake build is also supplied:

```bash
cmake -S models/native -B /tmp/neuralplsi-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/neuralplsi-build
/tmp/neuralplsi-build/neuralplsi_demo /tmp/example.nplsi
```

### Reproducibility and timing

The native RNG protocol is `splitmix64_v1`, with separate streams for weights,
validation splits and minibatch permutations. Fixed seeds reproduce native fits
without changing NumPy's global RNG state. The new protocol changes initialization
and splitting relative to the former Torch implementation, so old and new fits
need separate simulation directories. Simulation data generation itself is unchanged.

For a small timing check, including fit, prediction and native bootstrap refits (`--jobs 1` for serial):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/benchmark_backends.py --n 2000 --repeats 3 --bootstrap 24 --jobs 4
```

Despite its historical filename, this script now benchmarks only native C++ and
writes `output/benchmark_native.json`. Compilation/loading is reported separately.
These checks do not estimate coverage; run the full scenarios on HPC for that.

The historical timings and memory comparisons below used patience-based early
stopping and warm-start bootstrap refits, both since removed. They do not measure
the current training and bootstrap procedure.

Before the workspace changes, a local check used n=2,000, the `sfun` scenario,
data/fit seed 0, bootstrap seed 11,
24 warm-start refits and a 1,000-point link grid. Timings include resampling,
refits, grid evaluations, SEs and intervals, and exclude compilation and the
original point fit:

| Outcome | One C++ worker | Four C++ workers | Speedup |
|---|---:|---:|---:|
| Continuous | 1.136 s | 0.366 s | 3.10x |
| Binary | 0.933 s | 0.290 s | 3.22x |
| Cox | 1.755 s | 0.456 s | 3.85x |

All draws and summaries matched exactly across worker counts. These are single
local timing checks, not comparisons against the previous Python process pool.
Use `--bootstrap 24 --repeats 1 --bootstrap_seed 11 --g_grid_n 1000 --jobs 1`
and then `--jobs 4` with the timing script to repeat the configuration.

To reproduce the historical workspace comparison, run the following from a
checkout of commit `7270f3c`. Both compared revisions must use the same training
policy for its exact-result checks to be meaningful:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/compare_native_revisions.py --bootstrap 96 --repeats 3 --jobs 1 4
```

The script compiles commit `4637057` from the local git history with the current
compiler and compares it with the working tree. Compilation and data generation
are outside the timing intervals. It alternates the execution order and writes
raw timings and memory measurements to `output/native_revision_comparison.json`.
The standalone `reproduce/native_memory_check.cpp` measures peak process RSS
without Python. Its learning rate is zero to hold the weights fixed; it tests
allocations rather than estimation or convergence. Revisions before `aa80299`
stopped after 6 joint and 21 gamma epochs.
Use `--memory-n 0` to skip this check.

Local medians from three alternating runs used n=2,000, `sfun`, data/fit seed 0,
bootstrap seed 11, 96 warm-start refits and a 1,000-point grid. Bootstrap times
include refitting, grid evaluation and summaries, excluding the original fit:

| Outcome | Workers | Before | After | Before / after |
|---|---:|---:|---:|---:|
| Continuous | 1 | 3.446 s | 3.498 s | 0.99x |
| Continuous | 4 | 0.921 s | 0.921 s | 1.00x |
| Binary | 1 | 2.859 s | 2.724 s | 1.05x |
| Binary | 4 | 0.908 s | 0.764 s | 1.19x |
| Cox | 1 | 4.044 s | 3.976 s | 1.02x |
| Cox | 4 | 1.118 s | 1.102 s | 1.01x |

Runtime gains on this small network are modest and scenario dependent. The
larger benefit is memory. In the isolated Linux check with n=100,000, p=8,
q=3, two 32-unit hidden layers, batch size 64 and no link grid:

| Allocation check | Before peak RSS | After peak RSS |
|---|---:|---:|
| One point fit | 127.94 MiB | 12.33 MiB |
| Point fit followed by four bootstrap refits on four workers | 512.36 MiB | 31.43 MiB |

These include the input arrays and the complete native process. Linux RSS is
read from `/proc/self/status` (`VmHWM`) to exclude inherited pre-exec high-water
marks from the Python benchmark parent. The fixed-stopping memory check is
also available separately with `--memory-only`. Actual memory varies with the
data dimensions, architecture, bootstrap output and worker count.

## Corrected simulation and HPC workflow

Submit from the repository root. The array scripts default to the nine
NeuralPLSI scenarios. IDs 9–17 run the corresponding spline scenarios.
The scripts use the active Python environment unless a container is specified.
Existing cluster resource directives may need adjustment for another cluster.

```bash
mkdir -p logs
export NN_BACKEND=cpp
export N_BOOTSTRAP=999
export N_REPLICATES=500
export OUT_DIR=output/simulation
sbatch reproduce/run_simulation.slurm
sbatch reproduce/run_simulation_2000.slurm
# Include both estimators:
# sbatch --array=0-17 reproduce/run_simulation_2000.slurm
```

The default neural optimizer is now alternating. Use a fresh directory such as
`OUT_DIR=output/simulation_alternating`. `NN_OPTIMIZER=joint` selects the comparison
trainer, `NN_LINK_EPOCHS` sets the alternating block length, and
`NN_VALIDATION_FRACTION=0.2` enables checkpoint selection without early stopping.

To use the existing Singularity environment, set these before submission:

```bash
export NPLSI_PROJECT_DIR=/gpfs/scratch/doh03/NeuralPLSI
export NPLSI_CONTAINER=/gpfs/scratch/doh03/containers/nvidia-cuda12.9.sif
export NPLSI_OVERLAY=/gpfs/scratch/doh03/containers/survmix.ext3
export NPLSI_PYTHON=/ext3/miniconda3/bin/python
```

The C++ compiler must be available **inside** that container. Worker counts
respect the SLURM CPU allocation. `BOOTSTRAP_JOBS` controls native threads for
both estimators. BLAS/OpenMP threads are
limited to one. Reduce `BOOTSTRAP_JOBS` when memory is limited.
`SAVE_BOOTSTRAP_SAMPLES=1` retains beta/gamma draws in compressed NPZ files.
The default B remains 100 unless `N_BOOTSTRAP` is set. Increasing B reduces
bootstrap Monte Carlo error; it does not guarantee nominal coverage.
Results go to `output/simulation`, one file per estimator and scenario. Each
replicate records its run configuration, including a hash of the source code,
and a run refuses to resume into a file written by different code or settings;
remove the old results before rerunning after a code change. Full epoch budgets
increase runtime, so check the cluster wall-time limit before a full run.

For a local run:

```bash
python reproduce/main_simulation.py --models NeuralPLSI --nn_backend cpp \
  --device cpu --n_instances 2000 --g_fn sfun --outcome binary \
  --n_bootstrap 999 --bootstrap_jobs 8 --out output/simulation
```

Completed replicates are saved atomically. Failed replicates raise an error
with their seed, and concurrent writers to the same result file are rejected.
A run resumes only when its source hash, recorded software versions and settings
match. Use a fresh output directory after updating code or changing inference
settings. Do not combine old and corrected results. Backend and initialization
comparisons also belong in separate directories.

```bash
python reproduce/visualize_simulation.py --input output/simulation
python reproduce/table_simulation_comparison.py --input output/simulation
python reproduce/plot_inference.py --input output/simulation
```

`plot_inference.py` draws one figure that compares the two estimators:
interval coverage, SE calibration, RMSE and pointwise coverage of the g band.
The n = 2000 summaries in `results/simulation` predate alternating training
and use the historical joint optimizer with its 80/20 split.

Reports preserve sample size and exposure distribution, calculate coverage
from the saved interval endpoints, and provide per-coefficient bias, empirical
SD, average bootstrap SE, bootstrap centering bias and Monte Carlo SEs. Curve
plots use the recorded grid; their shading is the spread across simulation
replicates, not a bootstrap confidence band.

## Correctness changes

- Bootstrap refits always start from a fresh seeded initialization and are
  canonicalized to the same index orientation. New fits clear uncertainty
  estimates from previous fits. Saved model states use format version 3 and retain the optimizer,
  block length and holdout policy. Version 2 files load with their original
  joint-training settings; version 1 files are rejected.
- Cox likelihoods include every tied observation within the evaluated partition
  and use stable log-space calculations. Neural link minibatches still have
  batch-local risk sets; coefficient updates use all fitting rows. Cox intercepts are rejected because the
  partial likelihood cannot identify them. The link output bias receives only
  its identifying g(0) anchor gradient for Cox, avoiding numerical drift.
- Alternating training uses full-data coefficient steps and link-only AdamW.
  Both optimizers run their full neural epoch budgets. Optional validation
  selects complete checkpoints without early termination. Bootstrap refits
  use the same optimizer and training policy.
- Spline coefficients and knots correspond to the returned index: the final
  linear part is exactly the profile fit at that index. Profiling and final
  fitting use the same penalized objective. Serial and threaded bootstrap draws
  are identical.
- `initial=True` performs linear-model initialization using the fitting
  rows. It was previously ignored. It is still off by default.
- Simulation data retain the historical seeded draws without changing the
  caller's NumPy RNG state. Native initialization and minibatch shuffling use
  the versioned C++ RNG protocol. Corrected fits need fresh simulation results.
- NHANES preprocessing excludes incomplete observations explicitly; reported
  SEs use bootstrap sample SDs. Curve grids use the index, and curve-plus-intercept
  intervals include their joint bootstrap uncertainty. Plots preserve actual
  interval endpoints even when the point estimate lies outside the interval.

The default percentile interpolation remains `linear` (NumPy-compatible, now calculated in C++ for both estimators).
`--nn_bootstrap_quantile weibull` is available for a separate sensitivity run;
it changes endpoints, not point-estimator bias, and is not a coverage repair.
No graph penalty, score recalibration or experimental spline correction is
part of the production estimator. Spline coefficient intervals describe
coefficients in adaptively selected bases; use the bootstrap `g_grid` bands for
inference on the common link-function scale.

Run the regression tests with the command below. They independently verify
forward values and analytic gradients for every outcome/activation combination,
multi-step AdamW updates, linear initialization, full-budget/checkpoint behavior,
the exact gamma fit,
bootstrap orientation, serial/threaded reproducibility, native summaries and
serialization. Gradient checks include different layer depths and widths.
Workspace regressions compare lazy shuffling with an independently generated
explicit schedule through the full joint epoch budget and check chunk boundaries
and tied Cox validation losses. Training tests cover flat validation losses,
restoration after worsening validation losses, and the gamma fit: least squares
for continuous outcomes and a zero full-data gamma gradient for binary and Cox.
A bootstrap test blocks Python refits, RNG calls, process/thread creation and
percentile functions while asserting that exactly one native call completes the
whole operation. A separate subprocess blocks every Torch import.
Alternating tests independently differentiate a float64 network to check the
full coefficient gradient and Fisher/Gauss-Newton matrix, including explicit
global tied Cox risk sets. They also check frozen link weights during coefficient
steps, tangent step bounds, objective descent, link-only Adam clipping/decay,
complete checkpoint restoration, all-row initialization, and version-2 loading.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```
