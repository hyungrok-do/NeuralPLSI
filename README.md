# NeuralPLSI

Neural Partial-Linear Single Index (NeuralPLSI) is a method for modeling outcomes that depend on a linear combination of exposures (index) transformed by a flexible nonlinear function, adjusting for covariates.

Repository: [https://github.com/hyungrok-do/neuralplsi](https://github.com/hyungrok-do/neuralplsi)

Arxiv: [https://arxiv.org/abs/2512.11593](https://arxiv.org/abs/2512.11593)

## Overview

This package implements two models:
- **NeuralPLSI**: Uses a neural network to learn the nonlinear link function $g(\beta^T X)$.
- **PLSI**: Uses B-splines to estimate the link function (Wang et al., 2020).

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hyungrok-do/neuralplsi.git
cd neuralplsi
```

### 2. Install Dependencies
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

## C++ CPU training backend

The optional `cpp` backend runs the forward pass, analytic gradients, AdamW,
minibatches, validation, early stopping and gamma fine-tuning in C++17. The
single-index model, L2 normalization and bootstrap procedure are shared with the
Torch backend. The public Python API still uses PyTorch for initialization,
checkpoint objects and prediction. There are no LibTorch, Eigen or pybind11
build dependencies. Torch remains the default backend.

```python
from models import NeuralPLSI

model = NeuralPLSI(family="binary", activation="ELU", backend="cpp", device="cpu")
model.fit(X, Z, y, random_state=0)
boot = model.inference_bootstrap(X, Z, y, n_samples=999, n_jobs=8)
```

Linux/macOS with a C++17 compiler is required. `CXX` selects the compiler
(default `g++`). The library is compiled once, cached by source/compiler/platform,
and loaded through `ctypes`. Compilation is protected by a process lock so
bootstrap workers can share the cache. To prebuild inside your HPC environment:

```bash
python -m models.native
```

Use `NEURALPLSI_CACHE_DIR` to select a writable cache directory. C++ supports
continuous, binary and Cox outcomes, ELU, LeakyReLU, ReLU and Tanh, arbitrary
positive hidden widths/depths, and optional intercepts for continuous/binary
outcomes. It runs on CPU with float32 parameters. Unsupported configurations
raise an error. The native kernel does not silently fall back to Torch.

A local single-thread benchmark used n=2,000, p=8, q=3, the repository's `sfun`
scenario, 32 units in two hidden layers, seeds 0–2, and three serial warm-start
bootstrap refits per dataset. Median speedups across seeds were:

| Outcome | Fit | Bootstrap refits |
|---|---:|---:|
| Continuous | 17.5× | 18.2× |
| Binary | 17.1× | 21.8× |
| Cox | 16.6× | 20.4× |

Compilation/loading was excluded. Maximum absolute beta/gamma discrepancy,
including bootstrap draws, was 2.4e-7. Both backends selected the same stopping
epochs in these checks. These are small local timing checks, not a coverage
study or a promise of identical floating-point results on every machine.
Adam can amplify roundoff near a stationary point, particularly with an optional
linear initialization and full-batch training. Use the same backend throughout
a coverage study and record it with the results.
Reproduce the comparison with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/benchmark_backends.py --n 2000 --repeats 3
```

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
export OUT_DIR=output/simulation_fixed_cpp_b999
sbatch reproduce/run_simulation.slurm
sbatch reproduce/run_simulation_2000.slurm
# Include both estimators:
# sbatch --array=0-17 reproduce/run_simulation_2000.slurm
```

To use the existing Singularity environment, set these before submission:

```bash
export NPLSI_PROJECT_DIR=/gpfs/scratch/doh03/NeuralPLSI
export NPLSI_CONTAINER=/gpfs/scratch/doh03/containers/nvidia-cuda12.9.sif
export NPLSI_OVERLAY=/gpfs/scratch/doh03/containers/survmix.ext3
export NPLSI_PYTHON=/ext3/miniconda3/bin/python
```

The C++ compiler must be available **inside** that container. Worker counts
respect the SLURM CPU allocation and BLAS/OpenMP threads are limited to one per
worker. `BOOTSTRAP_JOBS` can reduce worker count when memory is limited.
`SAVE_BOOTSTRAP_SAMPLES=1` retains beta/gamma draws in compressed NPZ files.
The default B remains 100 unless `N_BOOTSTRAP` is set. Increasing B reduces
bootstrap Monte Carlo error; it does not guarantee nominal coverage.

For a local run:

```bash
python reproduce/main_simulation.py --models NeuralPLSI --nn_backend cpp \
  --device cpu --n_instances 2000 --g_fn sfun --outcome binary \
  --n_bootstrap 999 --bootstrap_jobs 8 --out output/simulation_fixed_cpp_b999
```

Completed replicates are saved atomically. Failed replicates raise an error
with their seed, and concurrent writers to the same result file are rejected.
A run resumes only when its source hash, recorded software versions and settings
match. Use a fresh output directory after updating code or changing inference
settings. Do not combine old and corrected results. Backend and initialization
comparisons also belong in separate directories.

```bash
python reproduce/visualize_simulation.py --input output/simulation_fixed_cpp_b999
python reproduce/table_simulation_comparison.py --input output/simulation_fixed_cpp_b999
```

Reports preserve sample size and exposure distribution, calculate coverage
from the saved interval endpoints, and provide per-coefficient bias, empirical
SD, average bootstrap SE, bootstrap centering bias and Monte Carlo SEs. Curve
plots use the recorded grid; their shading is the spread across simulation
replicates, not a bootstrap confidence band.

## Correctness changes

- Warm-start checkpoints preserve the index/link orientation. New fits clear
  uncertainty estimates from previous fits.
- Cox likelihoods use complete Breslow risk sets for ties and stable log-space
  calculations in both estimators. Cox intercepts are rejected because the
  partial likelihood cannot identify them. The link output bias receives only
  its identifying g(0) anchor gradient for Cox, avoiding numerical drift.
- Both stopping callbacks receive every validation loss. Training restores the
  best validation checkpoint. Gamma fine-tuning freezes the full link and clips
  only the parameters it updates.
- Spline coefficients and knots now correspond to the returned index. Binary
  profiling and final fitting use the same penalized objective. Serial and
  parallel bootstrap draws are reproducible.
- `initial=True` now performs linear-model initialization using the training
  partition. It was previously ignored. It is still off by default.
- Simulation data retain the historical seeded draws without changing the
  caller's NumPy RNG state. Torch and C++ training use the same local minibatch
  permutations. Corrected fits therefore need fresh simulation results.
- NHANES preprocessing excludes incomplete observations explicitly; reported
  SEs use bootstrap sample SDs. Curve grids use the index, and curve-plus-intercept
  intervals include their joint bootstrap uncertainty. Plots preserve actual
  interval endpoints even when the point estimate lies outside the interval.

The default percentile interpolation remains NumPy's `linear` method.
`--nn_bootstrap_quantile weibull` is available for a separate sensitivity run;
it changes endpoints, not point-estimator bias, and is not a coverage repair.
No graph penalty, score recalibration or experimental spline correction is
part of the production estimator. Spline coefficient intervals describe
coefficients in adaptively selected bases; use the bootstrap `g_grid` bands for
inference on the common link-function scale.

Run the regression and native parity checks with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```
