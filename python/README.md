# neuralplsi

Neural-network partial-linear single-index models for environmental mixtures:

    g(X @ beta) + Z @ gamma,   |beta| = 1

`X` holds the exposures, whose index `X @ beta` enters through a nonlinear link
`g`, and `Z` the covariates. Outcomes can be continuous, binary or
time-to-event (Cox). The package has two estimators:

- `NeuralPLSI`: `g` is a small neural network, fitted by full-batch L-BFGS.
- `SplinePLSI`: `g` is a cubic B-spline (Wang et al., 2020), the comparator.

Both fit and run the pairs bootstrap in a C++ engine
([`core/`](https://github.com/hyungrok-do/NeuralPLSI/blob/main/core/README.md)),
on native threads, which also computes NeuralPLSI predictions; SplinePLSI
predictions multiply the engine's spline basis by the coefficients in NumPy.
There is no Torch dependency. Paper: Do, Wang, Liu and Lee,
[Neural Network-based Partial-Linear Single-Index Models for Environmental
Mixtures Analysis](https://arxiv.org/abs/2512.11593), arXiv:2512.11593 (2025).

## Install

```bash
git clone https://github.com/hyungrok-do/NeuralPLSI.git
cd NeuralPLSI
python -m pip install .
```

This needs Python 3.10 or later, a C++17 compiler and CMake 3.16 or later, on
Linux or macOS. pip builds the engine with CMake (through scikit-build-core) and
installs the library inside the package, in `neuralplsi/_lib`. The runtime
dependencies are NumPy, pandas and SciPy. pip leaves its CMake build files in
`build/` in the checkout.

An editable install, `python -m pip install -e .`, also installs the CMake-built
library in `neuralplsi/_lib` in site-packages and uses the Python files of the
checkout. Changes to the Python files take effect without reinstalling; changes
to `core/` need `python -m pip install -e .` again.

```bash
python -m neuralplsi
```

prints the version and the library in use, for example
`neuralplsi 0.1.0: .../site-packages/neuralplsi/_lib/libneuralplsi.dylib`.

## Running from a checkout

The package also works from a source checkout without installation, for
example on a cluster whose Python environment is read-only. It then compiles
`core/` on first use with `$CXX` (default `g++`) and the flags
`-O3 -std=c++17 -shared -fPIC -ffp-contract=off -pthread`, into
`NEURALPLSI_CACHE_DIR` (default `~/.cache/neuralplsi`). The library is cached by
sources, compiler version, flags and platform, and a file lock lets parallel
jobs share one build. The scripts in
[`reproduce/`](https://github.com/hyungrok-do/NeuralPLSI/blob/main/reproduce/README.md)
and the HPC jobs use the package this way. To build the library ahead of the
jobs, from the repository root:

```bash
NEURALPLSI_CACHE_DIR=/path/to/cache PYTHONPATH=python python -m neuralplsi
```

`NEURALPLSI_LIBRARY` selects a prebuilt engine library instead, such as the
CMake build:

```bash
cmake -S core -B build && cmake --build build
NEURALPLSI_LIBRARY=build/libneuralplsi.dylib PYTHONPATH=python python -m neuralplsi
```

(`libneuralplsi.so` on Linux.) The library is chosen in this order:
`NEURALPLSI_LIBRARY`, which must name an existing file, the library installed
with the package, the checkout build. The engine version must equal the package
version.

## Quick start

```python
import numpy as np
from neuralplsi import NeuralPLSI, SplinePLSI, simulate_data

X, Z, y, *_ = simulate_data(1000, outcome='binary', g_type='sigmoid', seed=0)

model = NeuralPLSI(family='binary')
model.fit(X, Z, y, random_state=0)
print(model.beta, model.gamma)

grid = np.linspace(-3, 3, 61)
boot = model.inference_bootstrap(X, Z, y, n_samples=1000, random_state=0,
                                 g_grid=grid, n_jobs=8)
print(model.summary())          # estimates, bootstrap SEs, 95% percentile intervals
lower, upper = boot['g_lb'], boot['g_ub']   # pointwise 95% band for g on the grid

probability = model.predict_proba(X, Z)
model.save('exposure_model.nplsi')
restored = NeuralPLSI.load('exposure_model.nplsi')

spline = SplinePLSI(family='binary').fit(X, Z, y, random_state=0)
spline.inference_bootstrap(X, Z, y, n_samples=1000, random_state=0, g_grid=grid)
print(spline.summary())
```

`X` and `Z` are matrices with one row per observation; `Z` may have no
columns, and `Z=None` means no covariates. `y` is a vector, coded
0/1 for binary outcomes, or an `n x 2` array `[time, event]` for Cox outcomes,
with 0/1 events and at least one event. All values must be finite. NeuralPLSI
works in float32 and SplinePLSI in float64.

Predictions: `predict` returns the linear predictor, `predict_proba` the
probability for binary outcomes, `predict_partial_hazard` the partial hazard for
Cox outcomes, and `g_function(u)` the link at index values `u`. NeuralPLSI also
has `predict_gxb(X)`. `beta[0] >= 0` fixes the sign of the index.

`inference_bootstrap` resamples rows with replacement and refits each draw from
a fresh seeded initialization, in one call to the engine. It returns a
dictionary with the estimates `beta_hat` and `gamma_hat`, `n_workers`,
`bootstrap_rng` and the settings, and five entries for each block: the draws
(`beta_samples`), their mean (`beta_mean`), SE (`beta_se`, the sample SD) and
interval endpoints (`beta_lb`, `beta_ub`). The `beta` and `gamma` blocks are
always there. NeuralPLSI with `add_intercept=True` adds `intercept_hat` and the
`intercept` block; SplinePLSI adds `spline_hat` and the `spline` block (the
spline coefficients). Only when `g_grid` is given are there a `g` block, the
link on that grid, and `g_grid`. `summary()` then includes the SEs and
intervals; a new fit clears them. The draws do not depend on `n_jobs` or on the
number of draws: for the same `random_state`, the first B draws of a longer run
are identical. For the same `random_state`, both estimators are refitted on the
same resampled rows.

`inference_sandwich(X, Z, y, method='auto', ci=0.95, n_jobs='auto')` gives
standard errors and Wald intervals for beta, gamma and the intercept without
refitting: the variance of the minimizer of the fit's objective, `H^-1 C H^-1 /
n^2`, computed by the engine in double precision on the rows given (those the
model was fitted on), for continuous and binary outcomes. `method='full'` takes
the Hessian and score covariance over every weight of the network (dense, for
networks of at most 8,000 weights; `n_jobs` threads); `method='frozen'` over
beta, gamma and the intercept with the network held fixed (any size, slightly
narrower). `'auto'` is `'frozen'` for `optimizer='adam_frozen'` and `'full'`
otherwise. It returns the SEs and interval endpoints (`beta_se`, `beta_lb`,
`beta_ub`, ...) and, since the formula describes a minimizer, how far the fit
is from one: the largest beta and gamma components of the remaining Newton step
in SE units (`newton_beta`, `newton_gamma`; well below 1 at a converged fit) and
the Hessian's eigenvalue nearest zero (`min_eigenvalue`). `summary()` then
shows these SEs. `results/sandwich` has its coverage in simulation.

`save` writes the network weights, settings and index orientation to a text
file that round-trips float32 exactly; bootstrap results are not saved with it.
Fitted models can also be pickled.

## Parameters

`NeuralPLSI(family='continuous', max_iter=5000, weight_decay=1e-3, hidden_units=32,
n_hidden_layers=2, add_intercept=False, activation='ELU', initial=False,
optimizer='lbfgs', batch_size=64, learning_rate=1e-3, patience=10)`

| Parameter | Meaning |
|---|---|
| `family` | `'continuous'`, `'binary'` or `'cox'`. |
| `activation` | `'ELU'` (the default, used in the simulations and the NHANES analysis), `'LeakyReLU'`, `'ReLU'` or `'Tanh'`. |
| `hidden_units`, `n_hidden_layers` | Width and depth of the link network. |
| `weight_decay` | L2 penalty on the link network (not on beta, gamma or the intercept). |
| `max_iter` | L-BFGS iterations (AdamW epochs); 0 only initializes. |
| `add_intercept` | Intercept for continuous and binary outcomes. Cox models reject it. |
| `initial` | Start from a linear, logistic or Cox regression fitted to all rows. |
| `optimizer` | `'lbfgs'` (the default: full-batch L-BFGS to convergence), or minibatch AdamW with early stopping (`batch_size`, `learning_rate`, `patience`) followed on all rows by `'adam_exact_gamma'`, the exact gamma fit, or `'adam_frozen'`, the frozen correction of beta, gamma and the intercept. |

Each L-BFGS fit uses all rows: it minimizes the outcome loss plus a `g(0)^2`
anchor and the penalty until the objective converges (a relative decrease below
1e-9 over 20 iterations), then gamma (and the intercept) are fitted exactly by
Newton iterations. `fit(X, Z, y, random_state=0)`
seeds the initialization (an integer in `[0, 2**64)`). `device='cpu'` (or its
alias `'auto'`), `backend='cpp'` and `num_workers=0` are accepted for
compatibility and are the only supported values.

`inference_bootstrap(X, Z, y, n_samples=200, random_state=0, ci=0.95, g_grid=None,
n_jobs='auto', quantile_method='linear')` fits the model first if needed.
`n_jobs` limits the C++ threads: `'auto'` or -1 uses all available CPUs and -2
all but one, capped by CPU affinity, Linux cgroup quotas,
`SLURM_CPUS_PER_TASK` and `n_samples`. `quantile_method='weibull'` changes
the percentile interpolation, not the draws.

`SplinePLSI(family='continuous', num_knots=5, spline_degree=3, alpha=1e-6,
max_iter=200, tol=1e-6)` places `num_knots` knots at equally spaced order
statistics of the index, fits gamma and the spline coefficients exactly with
ridge penalty `alpha`, and profiles beta over the unit sphere by BFGS (at most
`max_iter` iterations, stopping when a step improves the loss by less than
`tol`). `fit(X, Z, y, beta_init=None, random_state=None)` starts from
`beta_init` or a random direction seeded by `random_state` (None draws the seed
from NumPy's global generator). Its `inference_bootstrap` takes the same
arguments except `quantile_method` (always linear), with `n_jobs=-1` by default.
For Cox outcomes, g is shifted to `g(0) = 0`. The fitted attributes are `beta`,
`gamma`, `spline_coeffs`, `knot_vector`, `loss_`, `n_iter_` and `converged_`.

The optimizer, the bootstrap protocol and their history are described in
[`core/README.md`](https://github.com/hyungrok-do/NeuralPLSI/blob/main/core/README.md).

## Simulation data

`neuralplsi.simulation` holds the paper's data-generating process.
`simulate_data(n, outcome='continuous', g_type='sigmoid', censoring_rate=0.3, seed=0, x_dist='normal')`
returns `X, Z, y, xb, gxb, g_fn`:

- `X`: 8 exposures, equicorrelated (0.3) normal, or uniform or t(5) with the
  same Gaussian copula (`x_dist='uniform'` or `'t'`).
- `Z`: two standard normal covariates and one +-1 covariate.
- `g_type`: `'linear'`, `'sigmoid'` or `'threshold'` (`2 tanh(1.5 max(u, 0)) + 0.1u`: nearly flat below 0, then a steep rise that levels off).
- `outcome`: `'continuous'` (normal errors), `'binary'` (logistic) or `'cox'`
  (exponential event times and exponential censoring).
- `xb`, `gxb` and `g_fn`: the true index, link values and link.

The true coefficients are `neuralplsi.simulation.beta` (unit length) and
`neuralplsi.simulation.gamma`. Data come from `numpy.random.RandomState(seed)`,
so the global NumPy random state is unchanged.

## Tests

```bash
python -m pip install ".[test]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest python/tests
```

Without the package installed, the tests use the checkout and compile `core/`
on first use. A plain `python -m pytest` from the repository root runs the same
tests; the tests of the reproduction scripts need `reproduce/requirements.txt`
and run with `python -m pytest reproduce/tests`.

The tests check forward values and analytic gradients against independent
NumPy references for every outcome and activation, and for several depths and
widths, the linear initialization, the iteration cap, convergence, the effect
of the penalty, the recorded penalized objective and the exact gamma fit (least
squares for continuous outcomes, a zero full-data gamma gradient for binary and
Cox). They check bootstrap orientation, serial and threaded reproducibility,
that each draw equals an explicit refit of its resampled rows, native summaries
against NumPy percentiles, Cox risk sets with ties, prediction chunk boundaries,
serialization, the AdamW optimizers and the sandwich, the spline fit and bootstrap, `Z=None`, the
library search and that the installed version, the package version and the
engine version agree. One test blocks Python refits, random number generation,
process and thread creation and percentile functions, and asserts that exactly
one native call completes the whole bootstrap.

## Tutorial

[`python/examples/tutorial.ipynb`](https://github.com/hyungrok-do/NeuralPLSI/blob/main/python/examples/tutorial.ipynb)
simulates continuous data, fits both models and plots the estimated links and
coefficients against the truth. It also needs matplotlib, which the package does
not install.

## License and citation

MIT ([LICENSE](https://github.com/hyungrok-do/NeuralPLSI/blob/main/LICENSE)).
Please cite:

```bibtex
@article{do2025neural,
  title={Neural Network-based Partial-Linear Single-Index Models for Environmental Mixtures Analysis},
  author={Do, Hyungrok and Wang, Yuyan and Liu, Mengling and Lee, Myeonggyun},
  journal={arXiv preprint arXiv:2512.11593},
  year={2025}
}
```

Wang, Y., Wu, Y., Jacobson, M.H. et al. A family of partial-linear single-index
models for analyzing complex environmental exposures with continuous,
categorical, time-to-event, and longitudinal health outcomes. *Environmental
Health* 19, 96 (2020). https://doi.org/10.1186/s12940-020-00644-4
