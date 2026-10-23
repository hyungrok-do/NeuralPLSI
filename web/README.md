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
`NeuralPLSI.load` in Python, `load_model()` in R and `nplsi_load` in C++). For
NHANES and CSV data a preprocessing file (`neuralplsi-preprocessing.json`) goes
with the model, which takes inputs as they were fitted: for each model column in
order, the file column it comes from, whether it was logged, the mean and SD it
was standardized with, and for indicators the level coded 1 and the reference.

A CSV file is read on the device. The page lists every column with its type
(numeric, 0/1, text or mostly missing), its missing and non-finite values and a
few examples, and the user gives each column one role: chemical exposure (X,
in the index inside g), confounder (Z, linear), outcome, event (0/1, Cox only) or
ignore, the default for IDs and survey weights. X and Z columns enter the model
in the order they were given their role (X rows show X1, X2, ...; β has a
positive weight on X1). Each X row has its own log box, with All and None for
the X rows shown; standardizing applies to every X, after its log. Text
columns can only be confounders and are dummy-coded against their first level.
A numeric confounder can be marked categorical and is then dummy-coded against
its lowest value; columns with 2 to 10 distinct integer values (other than
0/1) are suggested, not set. A continuous outcome can be logged and
standardized. Logs drop rows with a value of zero or below; numbers beyond
±3.4e38, the float32 range, count as non-finite. The outcome type follows a 0/1
outcome until the user sets it. A summary above Fit gives the outcome and
family, p and q (after dummy coding) and the rows used and dropped by reason;
Fit stays disabled, with the reason, until the setup can be fitted, including
networks over the engine's limit of 10,000,000 parameters. The roles and checks
are pure functions in `src/roles.js`.

With the roles and transforms of `reproduce/nhanes.py` (triglyceride logged
and standardized; its eight exposures in its order, logged except retinol,
standardized; age, and sex and race as categorical), the NHANES file opened as
a CSV gives the same X, Z and y as the built-in NHANES example, bit for bit,
and with ELU and the same seed the same fit.

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
`splsi_bootstrap` call; that a replicate whose allocation fails frees what it
allocated; the simulation design; CSV parsing; the column types, roles,
transforms and reasons of the CSV step (`roles.test.js`), including the NHANES
file set up as `reproduce/nhanes.py` against the built-in example; and, with a
Python that has numpy, pandas, scipy, scikit-learn and matplotlib, the NHANES
preprocessing against `reproduce/nhanes.py`, the bootstrap resamples against
the Python package, and that an exported model loads in `NeuralPLSI.load` and
predicts what the WebAssembly model predicts. The Python is the first of
`$PYTHON`, `python3`, `/usr/bin/python3` and `python` that imports those
packages; without one, those tests are skipped. `browser.test.js` drives the
built page in headless Chrome (set `CHROME` if it is not found;
`NPLSI_BROWSER=0` skips it), including the CSV step with a messy file, column
names that look like HTML (they must show as text), long names in the result
tables on phones 360, 390 and 430 px wide, the NHANES file as a user file
giving the same data and fit as the NHANES example, a failed NHANES download
and Retry, and the preprocessing download; it saves screenshots when
`NPLSI_SCREENSHOTS` names a directory.

## How the bootstrap runs

Each worker holds its own WebAssembly instance. Replicate b resamples rows with
`nplsi_bootstrap_indices(n, seed, b)`, imports the fitted model's state and
refits it with seed `seed + 1337 + b` (the `splitmix64_bootstrap_v1` protocol),
then returns β, γ, the intercept and g on the grid as one row. The page
collects the rows in replicate order and calls `nplsi_bootstrap_summary`.
SplinePLSI replicates use `splsi_fit` on the same resamples. Within one
WebAssembly build this equals the single-call bootstrap exactly, for any number
of workers. Against native builds, whose math library differs, LeakyReLU and
ReLU fits were identical in a check on simulated data, while ELU, Tanh and
SplinePLSI fits differed slightly, and single bootstrap replicates more than
point fits; the root [`README.md`](../README.md#one-engine) has the numbers.

## Deployment

`.github/workflows/web.yml` runs on pull requests and on pushes to `main` that
touch `core/`, `web/`, `NHANES/`, `python/neuralplsi/`, `reproduce/nhanes.py`
or the workflow, and on manual dispatch. It installs Emscripten 6.0.13 with
`emscripten-core/setup-emsdk` (v16, pinned by commit), builds and runs the
tests (without the browser test). Pushes and manual runs then publish
`web/dist` with `actions/deploy-pages`; only that job has the `pages: write`
and `id-token: write` permissions, and pull requests never deploy. Pages must
be set to deploy from GitHub Actions: Settings > Pages > Source: GitHub Actions.

The NHANES data are from the supplementary material of Wang et al. (2020),
*Environmental Health* 19:96, licensed CC BY 4.0.
