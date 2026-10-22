# Coefficient caching with updates every epoch

This compares the implementation in commit `114133d` with the coefficient-cache
optimization. Both use the alternating optimizer, all 2,000 supplied observations,
200 neural epochs, `link_epochs=1`, two hidden layers of 32 units, ELU, and the
same AdamW, sphere Gauss–Newton, Newton and backtracking settings.

Each usual coefficient cycle now needs two full-data network evaluations instead
of five: one produces both the link and its input derivative, and the other
evaluates the candidate beta. Gamma reuses these link values. Further beta
backtracking trials still evaluate the complete fitting loss. Fixed gamma data,
continuous-outcome curvature and Cox orderings are cached within each fit and
rebuilt for each bootstrap resample. Further rejected Newton trials avoid
gradient and Hessian calculations after the first rejection.

## Timing

Times are local wall-clock medians from three repeats, alternating which revision
runs first. All scenarios use `sfun`, data/fit seed 0, bootstrap seed 11, four cold
bootstrap refits and a 301-point link grid. Compilation and data generation are
excluded. Bootstrap timing excludes the original point fit and includes sampling,
refitting, link evaluation and interval calculation. The compiler was GCC 13.3.0
with `-O3 -ffp-contract=off`. BLAS/OpenMP thread counts were set to one. Timing
comparisons ran sequentially without concurrent test or simulation jobs from this
task. Results depend on the machine and its load.

Point-fit times below use the three runs paired with serial bootstrap:

| Outcome | Before | After | Time reduction |
|---|---:|---:|---:|
| Continuous | 3.3965 s | 1.9668 s | 42.1% |
| Binary | 3.2850 s | 2.0334 s | 38.1% |
| Cox | 3.3936 s | 2.0137 s | 40.7% |

Bootstrap times cover all four refits together:

| Outcome | C++ workers | Before | After | Time reduction |
|---|---:|---:|---:|---:|
| Continuous | 1 | 12.8705 s | 7.9326 s | 38.4% |
| Continuous | 4 | 4.9681 s | 3.1305 s | 37.0% |
| Binary | 1 | 12.8661 s | 8.1496 s | 36.7% |
| Binary | 4 | 5.4435 s | 3.3376 s | 38.7% |
| Cox | 1 | 13.2600 s | 8.1879 s | 38.3% |
| Cox | 4 | 5.9061 s | 3.1779 s | 46.2% |

These are repeated timings on one dataset per outcome, not new statistical
simulation results. Four refits suffice for this timing and equivalence check
but are not intended to estimate confidence-interval coverage.

## Numerical checks

All 18 before/after comparisons matched bit-for-bit for serialized model state
(including settings, weights and diagnostics), predictions, bootstrap draws and
returned array summaries. Each point fit completed 200 neural epochs and 200
accepted beta updates. Gamma accepted-step counts were unchanged: 401 continuous,
581 binary and 575 Cox.

An additional 62 cross-revision cases matched exactly. These covered all three
outcomes and four activations, held-out checkpoint selection, partial final
alternating cycles, the legacy joint optimizer with linear initialization, a
one-coordinate beta with skipped updates, no gamma covariates, optional
intercepts, tied Cox times, chunk boundaries and poor gamma starting values.
The 53-test suite passed, including an independent final-gamma score check after
accepted or skipped beta updates and threaded bootstrap comparisons against
explicit refits. No network-size, update-frequency or stopping-rule changes were
used to obtain the speedup.

Raw measurements and compatibility-case records are in
[coefficient_cache_benchmark.json](coefficient_cache_benchmark.json). The source
was benchmarked before its commit; the JSON records the parent revision and the
compiled-library SHA-256 identifying the tested binary.

## Reproduce

Run from the repository root, with `114133d` available in local git history:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/compare_native_revisions.py \
    --baseline-ref 114133d --n 2000 --epochs 200 --activation ELU \
    --optimizer alternating --validation-fraction 0 --link-epochs 1 \
    --bootstrap 4 --repeats 3 --jobs 1 4 --grid 301 --memory-n 0 \
    --out output/coefficient_cache_benchmark.json

python -m unittest discover -s tests -v
```
