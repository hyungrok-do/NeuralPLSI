# NeuralPLSI engine (C++)

The C++17 engine behind the `neuralplsi` Python and R packages and the web
demo. It fits two partial linear single-index models, `g(X @ beta) + Z @ gamma`,
with `|beta| = 1`:

- **NeuralPLSI** (`src/neuralplsi.cpp`): a neural-network link `g`, float32.
- **SplinePLSI** (`src/spline.cpp`): the B-spline comparator of Wang et al.
  (2020), double precision.

Initialization, the network, analytic backpropagation, the optimizers, the
exact gamma fit, prediction, serialization and the whole bootstrap (resampling,
refits, threads, SEs and intervals) run here. The engine uses only the C++
standard library and threads: no Torch or LibTorch, Eigen, BLAS or pybind11.
Every binding calls the plain C API in
[`include/neuralplsi.h`](include/neuralplsi.h): the Python package
([`python/`](../python/README.md)) through ctypes, the R package
([`r/`](../r/README.md)) through `.Call` on copies of these sources, and the
web demo ([`web/`](../web/README.md)) compiled to WebAssembly. SplinePLSI has
no prediction function: the bindings multiply the basis from `splsi_basis` by
the spline coefficients and add `Z gamma`.

| Path | Contents |
|---|---|
| `include/neuralplsi.h` | The C API. Every function, array layout and code is documented there. |
| `src/neuralplsi.cpp` | NeuralPLSI, its three optimizers, the bootstrap, the sandwich, worker counts and the state format. |
| `src/spline.cpp` | SplinePLSI. Its bootstrap reuses the NeuralPLSI resampling, worker and summary routines. |
| `examples/native_neuralplsi.cpp` | A C++ program that fits, bootstraps and saves a model. |
| `CMakeLists.txt` | Builds the shared library and the example, and registers the example as a test. |

## Build

With CMake (3.16 or later), from the repository root:

```bash
cmake -S core -B build
cmake --build build
ctest --test-dir build
```

This builds `libneuralplsi` (`.so` on Linux, `.dylib` on macOS) and
`neuralplsi_example`. The default build type is Release. `ctest` runs the
example as the test `example`. `-DNEURALPLSI_BUILD_EXAMPLES=OFF` builds the
library alone. The root `CMakeLists.txt` adds this directory; it is what
`pip install .` builds.

Without CMake, one command builds and runs the example:

```bash
g++ -O3 -std=c++17 -ffp-contract=off -pthread -I core/include \
  core/src/neuralplsi.cpp core/src/spline.cpp core/examples/native_neuralplsi.cpp \
  -o /tmp/nplsi-demo
/tmp/nplsi-demo /tmp/example.nplsi
```

The example trains a nonlinear model on 256 rows, runs an 8-draw bootstrap with
two threads and then one, checks that draws and summaries are identical and that
the source model is unchanged, and checks an exact save/load round trip.

The shared library, as the Python package compiles it from a checkout:

```bash
g++ -O3 -std=c++17 -shared -fPIC -ffp-contract=off -pthread -I core/include \
  core/src/neuralplsi.cpp core/src/spline.cpp -o /tmp/libneuralplsi.so
```

GCC and Clang builds use `-O3 -std=c++17 -ffp-contract=off`. Fused
multiply-add stays off, so the CMake build, the checkout build and the
pip-installed library produce the same float32 arithmetic on one machine. The R
package compiles at R's default optimization (`-O2` on macOS arm64), and its
`configure` adds `-ffp-contract=off` when the compiler accepts it; the web demo
uses `em++ -O3 -std=c++17 -ffp-contract=off`. Results on other CPUs, compilers
or math libraries can differ in the last bits; the root
[`README.md`](../README.md#one-engine) lists what was compared.

## C API

All matrices are contiguous, row-major float32 for NeuralPLSI and double for
SplinePLSI. Outcomes are `n` scalars, or `n x 2` `[time, event]` for Cox.
Functions return 0 on success and nonzero on failure; on failure `nplsi_create`,
`nplsi_import_state` and `nplsi_load` return a null handle, and
`nplsi_state_size` and `nplsi_bootstrap_workers` return 0. `nplsi_error()` and
`splsi_error()` give the message for the calling thread. `nplsi_version()`
returns the engine version, `"0.1.0"`, which bindings check against their own.

A model is created from two arrays:

- `config`, 13 `int32`: `p, q, width, hidden_layers, intercept, family,
  activation, parameter_count, max_iter, linear_init, optimizer, batch_size,
  patience`. family: continuous 0, binary 1, Cox 2. activation: ELU 0,
  LeakyReLU 1, ReLU 2, Tanh 3. optimizer: L-BFGS 0, AdamW then the exact gamma fit 2, AdamW then the
  frozen correction 3 (1, the plain AdamW baseline, was removed).
  `parameter_count = p + q + 1 + 2*width + (hidden_layers-1)*(width^2 + width) + width + 1 + intercept`.
  `batch_size` and `patience` must be at least 1 even for L-BFGS.
- `hyper`, 3 `float`: penalty, index sign (1 for a new model), AdamW learning
  rate. The learning rate must be positive even for L-BFGS.

| Functions | Purpose |
|---|---|
| `nplsi_create`, `nplsi_destroy`, `nplsi_version`, `nplsi_error` | Handles and errors. |
| `nplsi_fit` | Fit from a fresh initialization drawn from `seed`. |
| `nplsi_predict`, `nplsi_link`, `nplsi_coefficients` | Linear predictor, probability, partial hazard or `g(X beta)`; `g` on a grid; beta, gamma, intercept. |
| `nplsi_get_config`, `nplsi_get_weights`, `nplsi_set_weights`, `nplsi_canonicalize` | Settings, the four fit diagnostics and the raw parameter vector. |
| `nplsi_bootstrap`, `nplsi_bootstrap_workers`, `nplsi_bootstrap_indices`, `nplsi_bootstrap_summary` | The threaded pairs bootstrap and its parts. |
| `nplsi_sandwich` | Sandwich SEs of beta, gamma and the intercept, with the Newton-step diagnostic (Cox through Lin and Wei's score residuals): `full` over every weight (dense Hessian, at most 8,000 weights, threaded columns) or `frozen` over beta, gamma and the intercept with the network fixed (any size). |
| `nplsi_sandwich_gradient`, `nplsi_sandwich_scores`, `nplsi_sandwich_hessian` | The full sandwich's parts, in double precision: the objective's gradient, the rows' score sums and outer products, and Hessian columns by central differences (the web demo splits the columns across workers). |
| `nplsi_state_size`, `nplsi_export_state`, `nplsi_import_state`, `nplsi_save`, `nplsi_load` | Model state, in memory or in a file. |
| `nplsi_loss_grad`, `nplsi_outcome` | Unpenalized loss and gradients, for independent checks. |
| `splsi_fit`, `splsi_profile`, `splsi_basis`, `splsi_bootstrap`, `splsi_error` | SplinePLSI. |

Bootstrap `draws` are `B x D` doubles in replicate order, with columns beta,
gamma, intercept (NeuralPLSI) or spline coefficients (SplinePLSI), then `g` on
the grid. `summary` is `4 x D`: mean, sample SD (`ddof=1`), lower and upper
percentile endpoints. The caller owns both.

## NeuralPLSI

The index is `u = s * x'beta / |beta|`, where `s = +-1` is the index sign. The
link is `g(u) = MLP(u) + w*u`: a network with `hidden_layers` hidden layers of
`width` units and a linear skip weight `w`. The
linear predictor adds `z'gamma` and, for continuous and binary outcomes, an
optional intercept. A Cox partial likelihood cannot identify an intercept, so
Cox models reject one. Backpropagation loops over any depth and width. Network
parameters are float32; Cox risk-set sums, the gamma fit and the optional
linear initialization use double precision.

### Optimization

Each fit uses all rows, with no validation split, and minimizes

    outcome loss + g(0)^2 + (penalty / 2) * |link network|^2

by full-batch L-BFGS (memory 10, Armijo backtracking). The outcome loss is mean
squared error, binary cross-entropy, or the negative Breslow Cox partial
likelihood divided by the number of events, with complete risk sets for tied
times. The `g(0)^2` anchor identifies the level of the link. The penalty (1e-3
in the Python and R packages) covers the link-network weights and biases and
the skip weight, but not beta, gamma or the intercept. The loss does not depend
on the length of beta, so beta is renormalized to unit length after every step
instead of being penalized.

Training stops once the objective decreases by less than a relative 1e-5 over
10 iterations, when the gradient vanishes, or after `max_iter` iterations (5000
in the Python and R packages; `max_iter=0` performs initialization only).
Stopping at 1e-9 over 20 iterations took 3-11 times longer (10-33 times for a
bootstrap of 1,000 refits), changed beta and g RMSE by under 2% and sandwich
coverage by under 0.01; it removed the occasional threshold-g fit that stops
where the Hessian is nearly singular and gets a very wide sandwich SE, which the
sandwich's Newton-step diagnostic flags (`results/sandwich`). With
beta and the link fixed, gamma and the optional intercept are then fitted
exactly, with `g(x'beta)` as a fixed offset: least squares, logistic regression
or the Breslow Cox partial likelihood, by double-precision damped Newton
iterations. Original fits and bootstrap refits use the same procedure.

`linear_init=1` starts from a linear, logistic or Cox regression fitted to all
rows. It is off by default. The L2 index normalization, the skip weight and the
sign canonicalization (`beta[0] >= 0`, with the link reflected) are unchanged
from the earlier implementation.

This procedure replaced minibatch AdamW with an 80/20 training/validation split
and best-validation checkpoints, and, briefly, an alternating optimizer
(minibatch AdamW for the network with full-data Gauss-Newton beta and Newton
gamma steps after every epoch, commits `114133d` and `3fd7fdb`). The
alternating optimizer matched beta RMSE, but its network kept minibatch noise:
across fit seeds its g varied 2-7 times more than with L-BFGS, and fits took
1-9 times longer. At n = 2,000 the earlier procedure learned beta and g from
80% of the rows, which alone raised beta RMSE by about 12% (SplinePLSI fitted to
80% subsets lost the same), and its minibatch noise made fits from different
seeds disagree by half to four-fifths of the sampling SD of beta. Full-batch
fits are nearly deterministic: across seeds beta moves by about 0.0002-0.002,
against a sampling SD of 0.01-0.06. On 60 simulated datasets per setting, beta
RMSE matched SplinePLSI for continuous and Cox outcomes and was lower for binary
ones. The penalty replaces checkpoint selection as the regularizer. Its value,
1e-3, was chosen once from {3e-4, 1e-3, 3e-3, 1e-2} on those simulations and is
fixed, like the spline's five knots.

### AdamW

Optimizers 2 and 3 train like typical neural networks, for data or networks too
large for full-batch L-BFGS. AdamW (PyTorch's defaults: betas 0.9 and 0.999, eps
1e-8; learning rate `hyper[2]`; decoupled weight decay `hyper[0]` on the link
network) updates all parameters jointly on minibatches of `batch_size` rows
from a random 80% of the data. Training stops after `patience` epochs without
improvement in the loss on the other 20%, or after `max_iter` epochs, and
restores the best checkpoint. The simulations use at most 200 epochs, batch
size 64, weight decay 1e-4 and patience 10. Then, on all rows:

- `optimizer=2` (`NeuralPLSI-AdamGamma`) fits gamma and the optional intercept
  exactly, as L-BFGS does, with the link and index fixed.
- `optimizer=3` (`NeuralPLSI-AdamFrozen`) takes the frozen correction: damped
  Newton steps on beta (in the tangent plane of the unit sphere), gamma and the
  intercept with every network weight fixed, until a step moves no coordinate
  by more than 1e-8. These ~10 coordinates enter each row's linear predictor
  only through the index u and the linear terms, so the gradient and Hessian
  are exact from g(u), g'(u) and g''(u) (one forward pass carrying two
  derivatives in u) and the loss's derivatives in the linear predictor (for
  Cox, by cumulative sums over the risk sets): about three network passes per
  step at any network size. beta ends at unit length.

The diagnostics report the gamma iterations or the frozen correction's Newton
iterations; the objective stays the best validation loss. With `max_iter=0`
both only initialize.

## SplinePLSI

For each unit index beta, the knots (five in the Python and R packages) sit at
equally spaced order statistics of `x'beta` (Wang et al., 2020), and clamped
cubic B-splines give seven basis functions. Gamma and the spline coefficients
are the exact ridge-penalized fit on `[Z, B(x'beta)]` (alpha = 1e-6): least
squares by Cholesky, logistic regression and the Breslow Cox partial
likelihood by Newton.
Beta is profiled over the unit sphere by BFGS with central finite differences
(step 1e-3, which averages over the small kinks that order-statistic knots
create), followed by a compass search on the profile loss, from a seeded random
direction; `beta[0] >= 0` identifies its sign. A Cox likelihood cannot identify
the level of g, so the spline is shifted exactly to `g(0) = 0`; NeuralPLSI uses
the soft `g(0)^2` anchor instead. Outside the boundary knots, g is constant.

At the same beta, the C++ profile reproduces the former NumPy/SciPy
implementation (continuous: to machine precision; binary and Cox: equal or
lower loss, because Newton solves the inner fit exactly). On 20 simulated
n = 2,000 datasets per outcome, the full fits reached the same beta accuracy as
the SciPy version for continuous and binary outcomes, and fitted 30-100 times
faster for binary and Cox outcomes. The SciPy version's SLSQP search stopped
early on Cox outcomes (mean max |beta error| 0.31 against 0.033), which also
inflated its bootstrap SEs.

## Bootstrap and threads

`nplsi_bootstrap` and `splsi_bootstrap` run the whole pairs bootstrap in one
call, on native threads, with no callbacks. Each replicate `b`:

- resamples rows with `nplsi_bootstrap_indices(n, seed, b)`;
- refits from a fresh initialization (cold start) with fit seed
  `seed + 1337 + b`, so each draw re-samples the optimizer noise as well as the
  rows;
- canonicalizes the sign and evaluates `g` on the requested grid.

Each replicate writes its own output row, so the draws do not depend on the
number of threads, on scheduling, or on the total B: the first B draws of a
larger run are the draws of a B-draw run. The summaries depend only on the
draws. For the same seed, both
estimators are refitted on identical resamples. The source model supplies only
its configuration and is unchanged. Refitting the source state on the resampled
rows reproduces a draw exactly:
`NativeModel.from_bytes(source).fit(x[rows], z[rows], y[rows], random_state=seed+1337+b)`
in Python, checked in `python/tests/test_native_bootstrap.py` (L-BFGS) and
`python/tests/test_adam.py` (AdamW).

Warm starts from the original fit were removed: refits inherited its optimizer
error, and at n = 2,000 the bootstrap SE of beta was only 0.69-0.82 of its
sampling SD for binary and Cox outcomes (0.95 for continuous), against
1.00-1.04 with cold starts.

Means, sample SDs (`ddof=1`, centered before summing) and percentile endpoints
are computed in double precision. `quantile_method` 0 is linear interpolation
(type 7, NumPy's default) and 1 is Weibull (type 6). The choice changes the
endpoints only, not the draws or the estimator's bias.

`n_jobs` is a worker limit: -1 uses all available CPUs and -2 all but one. The
available count starts from `std::thread::hardware_concurrency()` and, on Linux,
is capped by the CPU affinity mask and cgroup v1/v2 CPU quotas, including
inherited ones; `SLURM_CPUS_PER_TASK` caps it on any system. Workers never
exceed B, and `workers_used` reports the count. Each fit is single-threaded. If
a refit fails, all threads are joined and the call fails with a message naming
the replicate and its fit seed; failed draws are never discarded or retried,
and all outputs must be discarded.

## Reproducibility

Native random numbers come from an explicit SplitMix64 generator with unbiased
bounded draws, so they are the same with every C++ standard library.

- `splitmix64_v1` initializes the network. L-BFGS draws only the initial
  weights; AdamW also draws its validation split and minibatch
  order, each from its own stream keyed by the fit seed.
- `splitmix64_bootstrap_v1` resamples: replicate `b` uses a stream keyed by
  `seed + b`, separate from the initialization streams, and fits with
  `seed + 1337 + b`. `seed + 1337 + B - 1` must fit in 64 bits.
- SplinePLSI draws its starting direction by SplitMix64 and Box-Muller from
  the fit seed (`seed + 1337 + b` in the bootstrap).

These protocols replaced the Torch and NumPy generators of earlier versions.
Fits and bootstrap draws from before the change are not comparable and need
separate result directories. Simulation data generation (in Python) is
unchanged.

## Model state

`nplsi_export_state` and `nplsi_save` write a portable text state, and
`nplsi_import_state` and `nplsi_load` read it. The format is version 5:

```text
NEURALPLSI_NATIVE 5
<13 config integers>
<3 hyper values, 9 significant digits>
<parameter_count weights, 9 significant digits>
<4 fit diagnostics, 17 significant digits>
```

It is written in the C locale and round-trips float32 exactly. The state holds
the weights, the settings (including the optimizer and its settings) and the
index orientation; bootstrap draws and intervals are separate results and are
not part of it. The diagnostics are the L-BFGS iterations (or AdamW epochs),
the gamma Newton iterations, the objective (for AdamW, the best validation
loss) and the restored AdamW epoch (0 for L-BFGS). Loading rejects other
versions: version 1 had the removed warm-start flag, version 2 the earlier
minibatch optimizer settings, version 3 the alternating optimizer's settings and
version 4 no optimizer choice. The state format version is separate from the
engine version (`nplsi_version()`).

## Memory

Each bootstrap worker reuses its training, gradient and inference buffers across
refits, and gathers each resampled dataset into one reusable buffer. The Cox
ordering and tie groups are prepared once per fit. Prediction and link-grid
evaluation use chunks of at most 256 rows and two reusable layer buffers. Input
data, cached link values and Cox risk-set storage grow with the number of
observations, so per-worker memory still matters when choosing `n_jobs`. The
bootstrap draws take `8 * B * D` bytes, shared by all workers, plus the
four-row summary.

Dense hidden layers pack up to 32 output neurons into a reusable tile, so the
compiler can vectorize across neurons. Each neuron's float32 accumulation order
is preserved, with fused multiply-add off. Each packing buffer holds at most
`32 * input_width` floats (4 KiB for a 32-unit hidden layer).

## Timings

The dense-layer packing reduced median point-fit time by 29-34% and bootstrap
time by 21-37% in a local check, with byte-identical fitted states, predictions
and bootstrap results. The settings, tables and raw measurements are in
[`results/dense_kernel_benchmark.md`](../results/dense_kernel_benchmark.md).
[`reproduce/README.md`](../reproduce/README.md) has the benchmark commands.

The timings and memory figures below used minibatch AdamW with patience-based
early stopping and warm-start bootstrap refits, all since removed. They do not
measure the current training and bootstrap procedure.

Before the workspace changes, a local check used n = 2,000, the `sfun`
scenario, data/fit seed 0, bootstrap seed 11, 24 warm-start refits and a
1,000-point link grid. Timings include resampling, refits, grid evaluations,
SEs and intervals, and exclude compilation and the original point fit:

| Outcome | One C++ worker | Four C++ workers | Speedup |
|---|---:|---:|---:|
| Continuous | 1.136 s | 0.366 s | 3.10x |
| Binary | 0.933 s | 0.290 s | 3.22x |
| Cox | 1.755 s | 0.456 s | 3.85x |

All draws and summaries matched exactly across worker counts. These are single
local timing checks, not comparisons against the former Python process pool.

The workspace changes were measured with `reproduce/compare_native_revisions.py`
at commit `7270f3c` (since removed). It compiled commit `4637057` and compared it
with the working tree, alternating the execution order, with compilation and
data generation outside the timed intervals. Local medians of three runs, with
n = 2,000, `sfun`, data/fit seed 0, bootstrap seed 11, 96 warm-start refits and
a 1,000-point grid, including refitting, grid evaluation and summaries and
excluding the original fit:

| Outcome | Workers | Before | After | Before / after |
|---|---:|---:|---:|---:|
| Continuous | 1 | 3.446 s | 3.498 s | 0.99x |
| Continuous | 4 | 0.921 s | 0.921 s | 1.00x |
| Binary | 1 | 2.859 s | 2.724 s | 1.05x |
| Binary | 4 | 0.908 s | 0.764 s | 1.19x |
| Cox | 1 | 4.044 s | 3.976 s | 1.02x |
| Cox | 4 | 1.118 s | 1.102 s | 1.01x |

Runtime gains on this small network were modest and scenario dependent. The
larger benefit was memory. At that commit, `reproduce/native_memory_check.cpp`
measured peak process RSS without Python, with a zero learning rate holding the
weights fixed (it now runs real full-batch fits). On Linux with n = 100,000,
p = 8, q = 3, two 32-unit hidden layers, batch size 64 and no link grid:

| Allocation check | Before peak RSS | After peak RSS |
|---|---:|---:|
| One point fit | 127.94 MiB | 12.33 MiB |
| Point fit followed by four bootstrap refits on four workers | 512.36 MiB | 31.43 MiB |

These include the input arrays and the whole native process. Linux RSS is read
from `/proc/self/status` (`VmHWM`), which excludes high-water marks inherited
from a Python parent. Actual memory varies with the data dimensions, the
architecture, the bootstrap output and the worker count.

## Correctness changes

- Bootstrap refits always start from a fresh seeded initialization and are
  canonicalized to the same index orientation. In the Python package, a new fit
  clears uncertainty estimates from a previous fit. Saved states use format
  version 5, which records the optimizer and its settings.
- Cox likelihoods use complete Breslow risk sets for ties and stable log-space
  sums in both estimators. Cox intercepts are rejected because the partial
  likelihood cannot identify them. For Cox outcomes the link's output bias
  receives only the gradient of its identifying `g(0)` anchor, avoiding
  numerical drift.
- The default fit trains on all rows by full-batch L-BFGS until the penalized
  objective converges. Gamma is then fitted exactly by Newton iterations with
  the index and link frozen. Minibatch AdamW with an 80/20 validation split and
  checkpoint selection is available only with a correction on all rows (the
  exact gamma fit or the frozen correction). Bootstrap refits use the same
  procedure as the original fit.
- Spline coefficients and knots correspond to the returned index: the final
  linear part is exactly the profile fit at that index. Profiling and the final
  fit use the same penalized objective. Serial and threaded bootstrap draws are
  identical.
- `linear_init=1` performs the linear-model initialization on all rows. It was
  previously ignored. It is still off by default.
- Percentile intervals are computed in C++ for both estimators. Linear
  interpolation remains the default; Weibull is available for sensitivity
  runs and is not a coverage repair. No graph penalty, score recalibration or
  experimental spline correction is part of either estimator. Spline coefficient
  intervals describe coefficients in adaptively placed bases; use the bootstrap
  bands of `g` on a grid for inference on the common link-function scale.
