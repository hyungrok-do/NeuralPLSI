# NeuralPLSI web demo

A static page that fits NeuralPLSI (and optionally SplinePLSI) with bootstrap
or sandwich inference in the browser. The C++ engine in `core/` is compiled to WebAssembly
and runs in Web Workers on the visitor's machine; data never leave the browser.
The page is hosted on GitHub Pages and works offline once loaded.

Data sources: your own data (a CSV file, the default) and the simulation
design of the paper (ported to JavaScript with a seeded generator; it does not
reproduce NumPy's draws). Under your own data, a **Use the NHANES example**
button opens the NHANES file as your data, set up as `reproduce/nhanes.py`
(`NHANES_SETUP` in `src/data.js`) with that script's activation and intercept;
like a file, it replaces the data open. Both sources go through the same column
setup below. The simulated data (columns x1-x8, z1-z3 and y, or time and event
for Cox) start set up as generated; the results show the true β, γ and g while
the simulated setup is the design. Each source keeps its own setup when the
visitor switches between them; a new simulated draw keeps the setup, a new
simulated outcome restarts it.

The steps are tabs along the top, each a full-width page: Data (the source,
then the column setup), Model, Run and Results; Next buttons lead from one to
the next, and the arrow keys, Home and End move between the tabs. The bar stays
in view with the progress of a run, Cancel and any error, and Results opens as
soon as a fit's estimates are drawn.

The Model page has the family (which follows the outcome type) and the seed,
shared by both models, then a card for NeuralPLSI and one for SplinePLSI side
by side, each with a **Run** box and its own settings: the network for
NeuralPLSI (activation, width, depth, weight decay, max iterations, intercept),
and knots, the ridge penalty and max iterations for SplinePLSI. Either model or
both can run; a model that does not run keeps its settings unchecked.

The main button, **Fit with bootstrap**, fits on all rows and then runs the
bootstrap; the estimates show as soon as the fit ends, with the standard errors
and intervals filled in when the bootstrap does. When the controls still match
the fit shown, it bootstraps that fit instead of fitting again. **Single fit**
gives the estimates alone. The Run page chooses the standard errors: the
bootstrap, the sandwich (continuous and binary outcomes) or both, with a
toggle in the results between the two.

Results: β and γ with bootstrap SEs and 95% percentile intervals, or sandwich
SEs and Wald intervals (estimate ± 1.96 SE), g(u) with bootstrap
pointwise bands, forest plots of β and γ with their intervals, fit diagnostics, and downloads of the
estimates, the bootstrap draws and the fitted model (`.nplsi`, read by
`NeuralPLSI.load` in Python, `load_model()` in R and `nplsi_load` in C++). For
NHANES and CSV data a preprocessing file (`neuralplsi-preprocessing.json`) goes
with the model, which takes inputs as they were fitted: for each model column in
order, the file column it comes from, whether it was logged, the mean and SD it
was standardized with, and for indicators the level coded 1 and the reference.

A CSV file is read on the device. The column setup has an outcome type and
four tabs: Columns (the role of each column), X (the chemical exposures in
model order, a Log-transform checkbox for all of them, mixed when only some are
logged, a log box for each and Standardize), Z (the confounders: each numeric
one has its own log, standardize and categorical boxes, with Log-transform and
Standardize boxes for all of them, set apart from the exposures'; a categorical
confounder has no log or z-score) and Outcome (the log and standardize boxes of
a continuous outcome). The arrow keys, Home and End move between the tabs.
Every numeric column has a small histogram of its values in the Columns tab;
in the X and Z tabs the histogram shows the values as the model takes them, on
the log scale when the column is logged.
The Columns tab lists every column with its type
(numeric, 0/1, text or mostly missing), its missing and non-finite values (with
the strings counted as missing) and a few examples, and the user gives each
column one role: chemical exposure (X,
in the index inside g), confounder (Z, linear), outcome, event (0/1, Cox only) or
ignore, the default for IDs and survey weights. X and Z columns enter the model
in the order they were given their role (X rows show X1, X2, ...; β has a
positive weight on X1). Standardizing applies to every X, after its log. Text
columns can only be confounders and are dummy-coded against their first level.
A numeric confounder can be marked categorical and is then dummy-coded against
its lowest value; columns with 2 to 10 distinct integer values (other than
0/1) are suggested, not set. A continuous outcome can be logged and
standardized. Logs drop rows with a value of zero or below; numbers beyond
±3.4e38, the float32 range, count as non-finite. The outcome type follows a 0/1
outcome until the user sets it. A summary above the run buttons gives the outcome and
family, p and q (after dummy coding) and the rows used and dropped by reason;
the buttons stay disabled, with the reason, until the setup can be fitted, including
networks over the engine's limit of 10,000,000 parameters. The roles and checks
are pure functions in `src/roles.js`. In the role table, the up and down arrows
move between columns without changing anything and the left and right arrows (or
Space) change the role; role changes, finished runs and bootstrap progress (at
25, 50 and 75%) are announced through one status line for screen readers.

Missing values follow pandas' defaults, matched exactly: an empty field, `NA`,
`N/A`, `n/a`, `NaN`, `nan`, `-NaN`, `-nan`, `NULL`, `null`, `None`, `<NA>`,
`#N/A`, `#N/A N/A`, `#NA`, `1.#IND`, `-1.#IND`, `1.#QNAN` and `-1.#QNAN`
(`NA_TOKENS` in `src/data.js`; a test checks them against pandas). In text
columns `None` is a category, not a missing value; `.` and `missing` are never
missing values, so a column of numbers that contains them is text, and its notes
name the values that are not numbers. A point is the decimal mark. A column of a
semicolon- or tab-separated file may use decimal commas (`1,5`) when none of its
values has a point. Its comma values decide: one with two or more groups
(`1,250,000`) can only have thousands separators; one that cannot be a
thousands number (`0,125`, `1,5`, `1234,5`) shows decimal commas for the whole
column; when every comma value could be either (`1,250`), the column is text.
A text column is flagged and cannot be X or the outcome. The examples show
the number read when it differs from the text (`1,5 → 1.5`). A binary outcome
that is all 0 or all 1 in the rows used cannot be fitted. Column names and
values are shown as text without their Unicode bidi controls, so a name cannot
reorder the text around it.

With the roles and transforms of `reproduce/nhanes.py` (triglyceride logged
and standardized; its eight exposures in its order, logged except retinol,
standardized; age, and sex and race as categorical), the NHANES file opened as
your own data and set up by hand gives the same X, Z and y as the NHANES
example, bit for bit, and with ELU and the same seed the same fit. Its file has
no missing or nonpositive values, so these match `reproduce/nhanes.py`, which
standardizes over each column's values rather than the rows used.

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
transforms and reasons of the column setup (`roles.test.js`), including the
NHANES setup and the simulated table, which set up as generated gives the
simulated data exactly; and, with a
Python that has numpy, pandas, scipy, scikit-learn and matplotlib, the NHANES
setup against `reproduce/nhanes.py`, the bootstrap resamples against
the Python package, and that an exported model loads in `NeuralPLSI.load` and
predicts what the WebAssembly model predicts. The Python is the first of
`$PYTHON`, `python3`, `/usr/bin/python3` and `python` that imports those
packages; without one, those tests are skipped. `browser.test.js` drives the
built page in headless Chrome (set `CHROME` if it is not found;
`NPLSI_BROWSER=0` skips it), including Fit with bootstrap and Single fit, a
bootstrap of the fit shown, sandwich standard errors (off for Cox, and the Bootstrap/Sandwich toggle), the simulated data in the column setup (the tabs,
the Log-transform checkbox and when the truth applies), the CSV step with a messy file, column
names that look like HTML (they must show as text), long names in the result
tables on phones 360, 390 and 430 px wide, the NHANES file as a user file
giving the same data and fit as the NHANES example, a failed NHANES download
and Retry, the preprocessing download, the role table on a 360 px phone in Cox
mode, missing-value notes and decimal commas, text contrast in light and dark,
focus and announcements around the run buttons and Cancel, arrow keys in the role table,
Cancel while the WebAssembly module is still loading, a second file chosen
before the first is read, and bidi controls in a column name; it saves
screenshots when `NPLSI_SCREENSHOTS` names a directory. CI installs the Python
packages pinned in `tests/requirements.txt`.

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

## How the sandwich runs

The sandwich is the M-estimator variance of the fitted minimizer,
H⁻¹CH⁻¹/n² for the penalized objective, over every weight of the network
(β in the tangent plane of the unit sphere), as in
[`results/sandwich`](../results/sandwich/README.md). The engine gives, in double
precision, the objective's gradient (`nplsi_sandwich_gradient`), the sum of the
rows' score vectors and of their outer products (`nplsi_sandwich_scores`) and
columns of the Hessian by central differences of the exact gradient
(`nplsi_sandwich_hessian`); the page splits the Hessian's columns across the
workers and assembles the parts in `src/sandwich.js` (LU with partial pivoting).
SplinePLSI's sandwich, with its knots held fixed, is computed in `src/sandwich.js`
from analytic derivatives. Both match `reproduce/sandwich/sandwich.py` and
`spline_sandwich.py` on simulated fits: β SEs within 1e-4 relative (NeuralPLSI)
and 1e-14 (SplinePLSI). The results report the Newton step that remains (in SEs
of β and γ) and the Hessian's eigenvalue nearest zero, and warn when the step
exceeds 0.1 SE or the Hessian is indefinite: the sandwich describes a minimizer,
which the engine's L-BFGS, run to convergence, reaches. At n = 500 with 32×2
hidden units (about 1,200 weights) it takes about 3 s on 9 workers.

## Deployment

`.github/workflows/web.yml` runs on pull requests and on pushes to `main` that
touch `core/`, `web/`, `NHANES/`, `python/neuralplsi/`, `reproduce/nhanes.py`
or the workflow, and on manual dispatch. It installs Emscripten 6.0.13 with
`emscripten-core/setup-emsdk` (v16, pinned by commit) and the Python packages
of `tests/requirements.txt`, builds and runs the tests, then the browser test in
its own step with the runner's Chrome (`CHROME=/usr/bin/google-chrome`). The
checkout does not keep its token. Pushes to `main`, and manual runs on `main`,
then publish `web/dist` with `actions/deploy-pages`; only that job has the
`pages: write` and `id-token: write` permissions, and pull requests and other
branches never deploy. Pages must
be set to deploy from GitHub Actions: Settings > Pages > Source: GitHub Actions.

The NHANES data are from the supplementary material of Wang et al. (2020),
*Environmental Health* 19:96, licensed CC BY 4.0.
