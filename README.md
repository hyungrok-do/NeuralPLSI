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
backpropagation, AdamW, data splitting, minibatch shuffling, early stopping,
prediction and model serialization all run in C++. There is **no Torch or
LibTorch dependency**, including in the Python examples and tests. The Python API
uses NumPy arrays and handles bootstrap orchestration and reporting. The separate
SplinePLSI comparator continues to use NumPy/SciPy.

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
compiler and platform, and loaded through `ctypes`. A process lock coordinates
bootstrap workers. No Eigen, pybind11 or external numerical library is needed
by the C++ model. To prebuild inside your HPC environment:

```bash
python -m models.native
```

Use `NEURALPLSI_CACHE_DIR` to select a writable cache directory. The implementation
supports continuous, binary and Cox outcomes, ELU, LeakyReLU, ReLU and Tanh, and
optional intercepts for continuous/binary outcomes. It uses float32 network
parameters and double precision for Cox risk-set accumulation and the optional
linear initialization. `backend="cpp"` and `device="cpu"` are the defaults and
only supported execution modes (`device="auto"` is a CPU alias).

### Optimization

Training uses **minibatch AdamW**, with explicit C++ backpropagation. It uses
stochastic minibatches, with Adam's adaptive first and second moments and
decoupled weight decay. Default settings are:

| Parameter group | Joint training learning rate | Fine-tuning learning rate | Weight decay |
|---|---:|---:|---:|
| Raw index weights, beta | 0.001 | Frozen | 0.00001 |
| Link network and linear skip | 0.001 | Frozen | 0.0001 |
| Covariate coefficients, gamma, and optional intercept | 0.05 | 0.01 | 0 |

The `learning_rate` argument controls the base rate. Gamma uses 50 times that
rate during joint training and 10 times that rate during fine-tuning. Adam's
moment coefficients are 0.9 and 0.999, with epsilon 1e-8. The default gradient
norm limit is 1.0; fine-tuning clips only gamma and the optional intercept.

Each fit uses an 80/20 training/validation split and shuffled minibatches of 64.
Joint training runs for at most 200 epochs, with patience 5. The best joint
checkpoint is restored before freezing beta and the entire link. Gamma then
trains for at most `min(max_epoch, 50)` epochs, with patience 20, and its best
validation checkpoint is restored. Gamma fine-tuning starts with fresh Adam
moments. `max_epoch=0` performs initialization only.

The objectives are mean squared error, binary cross-entropy, or the negative
Breslow Cox partial likelihood divided by the event count, plus the identifying
`g(0)^2` anchor during joint training. Cox risk sets include all tied observations
within each training batch; validation uses its full partition. The original L2
index normalization, linear skip and sign canonicalization are retained.
`initial=True` optionally starts from a native linear/logistic/Cox regression
fit to the training partition; it is off by default.

### Use without Python

The C API is declared in `models/native/core.h`. The example trains a nonlinear
model and checks a native save/load round trip using only the C++ standard library:

```bash
g++ -O3 -ffp-contract=off -std=c++17 -I models/native \
  models/native/core.cpp examples/native_neuralplsi.cpp -o /tmp/neuralplsi_demo
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

For a small timing check, including fit, prediction and serial bootstrap refits:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/benchmark_backends.py --n 2000 --repeats 3
```

Despite its historical filename, this script now benchmarks only native C++ and
writes `output/benchmark_native.json`. Compilation/loading is reported separately.
These checks do not estimate coverage; run the full scenarios on HPC for that.

A local single-thread check with n=2,000, p=8, q=3, the `sfun` scenario,
two hidden layers of 32 ELU units, and seeds 0–2 gave these median wall times:

| Outcome | Fit | Predict 2,000 rows | Three warm-start bootstrap refits |
|---|---:|---:|---:|
| Continuous | 0.079 s | 0.0020 s | 0.143 s |
| Binary | 0.111 s | 0.0020 s | 0.134 s |
| Cox | 0.097 s | 0.0020 s | 0.114 s |

These measurements exclude compilation/loading and are specific to this machine
and early-stopping behavior. They are not a comparison against the old RNG protocol.

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
export OUT_DIR=output/simulation_native_v3_b999
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
  --n_bootstrap 999 --bootstrap_jobs 8 --out output/simulation_native_v3_b999
```

Completed replicates are saved atomically. Failed replicates raise an error
with their seed, and concurrent writers to the same result file are rejected.
A run resumes only when its source hash, recorded software versions and settings
match. Use a fresh output directory after updating code or changing inference
settings. Do not combine old and corrected results. Backend and initialization
comparisons also belong in separate directories.

```bash
python reproduce/visualize_simulation.py --input output/simulation_native_v3_b999
python reproduce/table_simulation_comparison.py --input output/simulation_native_v3_b999
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
  caller's NumPy RNG state. Native initialization and minibatch shuffling use
  the versioned C++ RNG protocol. Corrected fits need fresh simulation results.
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

Run the regression tests with the command below. They independently verify
forward values and analytic gradients for every outcome/activation combination,
multi-step AdamW updates, linear initialization, stopping/checkpoint behavior,
bootstrap orientation, serial/parallel reproducibility, and native serialization.
A separate subprocess also blocks every Torch import during fitting and inference.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```
