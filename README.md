# NeuralPLSI

NeuralPLSI fits partial-linear single-index models for environmental mixtures:

    g(X'beta) + Z'gamma,   |beta| = 1

The exposures `X` enter through a single index `X'beta`, whose weights `beta`
give the relative contribution of each exposure, and a nonlinear link `g`,
here a small neural network fitted by full-batch L-BFGS. The covariates `Z`
enter linearly. Outcomes can be continuous, binary or time-to-event (Cox). A
pairs bootstrap gives standard errors and percentile intervals for `beta` and
`gamma` and pointwise bands for `g`. SplinePLSI, the B-spline model of Wang et
al. (2020), is included as the comparator. Paper: Do, Wang, Liu and Lee,
[Neural Network-based Partial-Linear Single-Index Models for Environmental
Mixtures Analysis](https://arxiv.org/abs/2512.11593), arXiv:2512.11593 (2025).

## Repository

| Path | Contents |
|---|---|
| [`core/`](core/README.md) | The C++17 engine and its plain C API: both models, the bootstrap and model files. |
| [`python/`](python/README.md) | The Python package `neuralplsi`, a ctypes binding, with its tests and a tutorial. |
| [`r/`](r/README.md) | The R package `neuralplsi`, a `.Call` binding. |
| [`web/`](web/README.md) | A browser demo: the engine compiled to WebAssembly. |
| [`reproduce/`](reproduce/README.md) | The paper's simulations, NHANES analysis, HPC jobs, reports and benchmarks. |
| [`results/`](results/) | Committed summaries of the simulation study, the NHANES analysis and a benchmark. |
| `NHANES/` | The NHANES data of Wang et al. (2020). |

Each README has the details: parameters, the optimizer, the bootstrap
protocol, the model file format and the history of the implementation.

## Quick start

### Python

Needs Python 3.10 or later, a C++17 compiler and CMake 3.16 or later.

```bash
git clone https://github.com/hyungrok-do/NeuralPLSI.git
cd NeuralPLSI
python -m pip install .
```

```python
from neuralplsi import NeuralPLSI, simulate_data

X, Z, y, *_ = simulate_data(1000, outcome='binary', g_type='sigmoid', seed=0)
model = NeuralPLSI(family='binary').fit(X, Z, y, random_state=0)
model.inference_bootstrap(X, Z, y, n_samples=1000, random_state=0)
print(model.summary())
```

The 1000 bootstrap refits run on native threads and took about 20 seconds on a
10-core laptop.

### R

Needs a C++17 compiler. From the repository root:

```bash
R CMD INSTALL r
```

```r
library(neuralplsi)
set.seed(1)
n <- 1000
x <- matrix(rnorm(6 * n), n, 6) * sqrt(0.7) + rnorm(n) * sqrt(0.3)
colnames(x) <- c("lead", "cadmium", "mercury", "arsenic", "pfoa", "bpa")
z <- cbind(age = rnorm(n), sex = rbinom(n, 1, 0.5))
beta <- c(0.5, 0.5, -0.5, 0.4, 0.3, 0)
g <- function(u) 5 * (plogis(2 * u) - 0.5)
y <- g(drop(x %*% beta)) + 0.5 * z[, "age"] - 0.3 * z[, "sex"] + rnorm(n)

fit <- plsi_bootstrap(neuralplsi(x, z, y), B = 1000, seed = 0)
summary(fit)
```

This is the sigmoid design of the paper's simulations with six correlated
exposures. The 1000 bootstrap refits took about 50 seconds on a 10-core laptop;
[`r/README.md`](r/README.md) shows the summary.

### C++

```bash
cmake -S core -B build
cmake --build build
ctest --test-dir build
```

This builds `libneuralplsi` and runs
[`core/examples/native_neuralplsi.cpp`](core/examples/native_neuralplsi.cpp),
which fits, bootstraps, saves and reloads a model through the C API in
[`core/include/neuralplsi.h`](core/include/neuralplsi.h).

### Web demo

https://hyungrok-do.github.io/NeuralPLSI/ fits NeuralPLSI, SplinePLSI or both with
bootstrap inference in the browser, on your own CSV file (or the NHANES example,
one button away) or simulated data; the data stay on the visitor's machine.
Both go through the same column setup: the user gives each column a role
(exposure X, confounder Z, outcome, event for Cox outcomes, or ignore), with a
small histogram of each column, and tabs for X, Z and the outcome set the
transforms. Exposures and numeric confounders can each be logged and
standardized, a numeric confounder can be treated as categorical, and a
continuous outcome can be logged and standardized. The NHANES example opens set
up as `reproduce/nhanes.py`. Fit with bootstrap is the main run; a single fit is the
quick alternative.
[`web/README.md`](web/README.md) has the details. The page is live once GitHub
Pages is enabled for the repository (Settings > Pages > Source: GitHub
Actions). To build and serve it locally, with Emscripten and Node:

```bash
web/build.sh
python3 -m http.server -d web/dist
```

and open http://localhost:8000.

## One engine

Every binding calls the same C API. Fitting, the bootstrap, NeuralPLSI
predictions and model files run in the engine. SplinePLSI predictions are
`Z gamma` plus the engine's B-spline basis (`splsi_basis`) times the spline
coefficients, computed in each language. Python loads the engine library
through ctypes. The R package compiles copies of the engine's two sources and
its header, kept identical to `core/` by `r/tools/sync-core.sh`, and calls them
through `.Call`. The web demo compiles the sources in `core/` to WebAssembly.
All of them follow the same bootstrap protocol: replicate `b` resamples rows
with seed `seed + b` and refits from a fresh initialization with seed
`seed + 1337 + b`, so the draws do not depend on the number of threads or on B.

Every GCC, Clang and Emscripten build compiles with `-ffp-contract=off`, so the
compiler does not fuse multiply-adds. On macOS arm64 with Apple clang, the
pip-installed library, the CMake build, the library the Python package compiles
from a checkout and the R package (built at `-O2`) gave identical fits,
bootstrap draws and NeuralPLSI predictions for the same data and seeds. Model
files (`.nplsi`) written by the C++ example load in Python and R, and Python
and R read each other's. SplinePLSI predictions and `g` values agree to
rounding error (a few times 1e-15 in the R package's comparison with Python),
because NumPy and R sum the final products in different orders. Other CPUs,
compilers or math libraries can give slightly different fits.

The WebAssembly build uses Emscripten's math library, which rounds some results
of `exp`, `expm1`, `log1p` and `tanh` differently from the native one. In a
check on simulated data (n = 1000, sigmoid g, seed 0, all three outcomes),
LeakyReLU and ReLU fits, whose networks call none of these functions, were
identical to the native fits, and so were 20 LeakyReLU bootstrap replicates.
ELU and Tanh fits differed by up to 0.003 in `beta` (0.0015 for the binary
Python example above), and SplinePLSI fits by at most 1e-9. Single bootstrap
replicates can differ more than point fits: by up to 0.009 in `beta` for ELU,
and by up to 0.004 for SplinePLSI with a binary outcome.

## Reproducing the paper

[`reproduce/README.md`](reproduce/README.md) describes the simulation study, the
HPC jobs, the reports and the NHANES analysis. The summaries of the current
n = 2000 simulation run are in [`results/simulation`](results/simulation/README.md)
and the NHANES results in [`results/nhanes`](results/nhanes/README.md).

`NHANES/12940_2020_644_MOESM2_ESM.csv` is the NHANES dataset from the
[supplementary material](https://static-content.springer.com/esm/art%3A10.1186%2Fs12940-020-00644-4/MediaObjects/12940_2020_644_MOESM2_ESM.csv)
of Wang et al. (2020), open access under CC BY 4.0.

## Citation

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

## License

MIT. See [LICENSE](LICENSE).
