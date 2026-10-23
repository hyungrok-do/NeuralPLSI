# NeuralPLSI web demo

A static page that fits NeuralPLSI (and optionally SplinePLSI) with bootstrap
inference in the browser. The C++ engine in `core/` is compiled to WebAssembly
and runs in Web Workers on the visitor's machine; data never leave the browser.
The page is hosted on GitHub Pages and works offline once loaded.

Data sources: the simulation design of the paper (ported to JavaScript with a
seeded generator; it does not reproduce NumPy's draws), the NHANES example
prepared as in `reproduce/nhanes.py`, and a CSV file chosen by the user.
Results: β and γ with bootstrap SEs and 95% percentile intervals, g(u) with
pointwise bands, a forest plot of β, fit diagnostics, and downloads of the
estimates, the bootstrap draws and the fitted model (`.nplsi`, read by
`NeuralPLSI.load` in Python and `nplsi_load` in C++).

## Build

Requires Emscripten and Node.

```bash
brew install emscripten
web/build.sh
```

`build.sh` compiles `core/src/neuralplsi.cpp` and `core/src/spline.cpp` with
`em++ -O3 -std=c++17 -ffp-contract=off -fwasm-exceptions` and copies the page,
its modules and the NHANES file into `web/dist/`. Homebrew's Emscripten has no
configuration file; the script then writes one to `web/.cache/` that points at
the LLVM and Binaryen Homebrew installed. The build has no pthreads: GitHub
Pages cannot send the headers that SharedArrayBuffer needs.

## Serve locally

```bash
python3 -m http.server -d web/dist
```

and open http://localhost:8000. Module workers and the service worker need
http(s); opening `index.html` as a file does not work.

## Test

```bash
node --test web/tests/*.test.js
```

The tests need a build in `web/dist` and no npm packages. They check that the
engine loads and that fits lower the objective for all three families; that
the page's per-replicate bootstrap, run on two instances in any order, gives
the same draws and summary, byte for byte, as one `nplsi_bootstrap` or
`splsi_bootstrap` call; the simulation design; CSV parsing; and, with a Python
that has numpy, pandas, scipy, scikit-learn and matplotlib (`$PYTHON`, else
`python3`, else `/usr/bin/python3`), the NHANES preprocessing against
`reproduce/nhanes.py`, the bootstrap resamples against the Python package, and
that an exported model loads in `NeuralPLSI.load` and predicts what the
WebAssembly model predicts. `browser.test.js` drives the built page in headless
Chrome (set `CHROME` if it is not found; `NPLSI_BROWSER=0` skips it) and saves
screenshots when `NPLSI_SCREENSHOTS` names a directory.

## How the bootstrap runs

Each worker holds its own WebAssembly instance. Replicate b resamples rows with
`nplsi_bootstrap_indices(n, seed, b)`, imports the fitted model's state and
refits it with seed `seed + 1337 + b` (the `splitmix64_bootstrap_v1` protocol),
then returns β, γ, the intercept and g on the grid as one row. The page
collects the rows in replicate order and calls `nplsi_bootstrap_summary`.
SplinePLSI replicates use `splsi_fit` on the same resamples. Within one
WebAssembly build this equals the single-call bootstrap exactly, for any number
of workers. WebAssembly results are not bit-identical to native builds, whose
math library differs.

## Deployment

`.github/workflows/web.yml` runs on pushes to `main` that touch `core/`, `web/`,
`NHANES/` or the workflow, and on manual dispatch. It installs Emscripten
6.0.13 with `mymindstorm/setup-emsdk`, builds, runs the tests (without the
browser test) and publishes `web/dist` with `actions/deploy-pages`. Pages must
be set to deploy from GitHub Actions: Settings > Pages > Source: GitHub Actions.

The NHANES data are from the supplementary material of Wang et al. (2020),
*Environmental Health* 19:96, licensed CC BY 4.0.
