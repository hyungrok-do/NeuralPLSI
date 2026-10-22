# Dense-layer implementation optimization

The packed dense-layer kernel reduced median point-fit time by **29–34%** and
bootstrap time by **21–37%** in this local check. This is an implementation
change to the current full-batch L-BFGS estimator. The network remains two
hidden layers of 32 units, with the same objective, penalty, convergence rules,
gamma fit, initialization and cold-start bootstrap procedure.

The baseline is `b74333674c63adee16b61a64f4aad9240a78a56b`, which already includes
full-batch L-BFGS and the refreshed NHANES results. The later NHANES curve export
in `6d391f3` was preserved; it does not change the model code. Earlier timings
against the alternating optimizer do not describe this comparison.

The kernel packs at most 32 output neurons so the compiler can vectorize across
independent outputs. Each neuron's float32 multiplication/addition order is
preserved. Scalar-input and narrow-output layers retain their scalar path.
The compiler flags retain `-ffp-contract=off`, without fast-math or
architecture-specific instructions. The packing tile is reusable and bounded
by `32 * input_width` floats per workspace, or 4 KiB at the default hidden width.

Timings use n = 2,000, `sfun`, ELU activation, data/fit seed 0, bootstrap seed 11,
four bootstrap draws, a 301-point link grid and the unchanged 5,000-iteration cap.
Three repeats alternate baseline/optimized execution order for each outcome and
worker count. Compilation and data generation are outside the timed intervals.
Both versions run in the same process with the same compiler (g++ 13.3.0) on an
AMD EPYC 9V74 CPU, with BLAS/OpenMP threads limited to one. The raw observations,
source hashes and compiler configuration are in [dense_kernel_benchmark.json](dense_kernel_benchmark.json).

Point-fit medians pool all six measurements per version/outcome (three repeats
in each bootstrap-worker setting; the original fit itself is single-threaded).

| Outcome | Before | After | Time reduction | L-BFGS iterations, both versions |
|---|---:|---:|---:|---:|
| Continuous | 1.280 s | 0.911 s | 28.8% | 334 |
| Binary | 0.901 s | 0.627 s | 30.5% | 220 |
| Cox | 0.780 s | 0.518 s | 33.5% | 197 |

Bootstrap medians use three measurements per row and include all four refits,
resampling, grid evaluation, SEs and intervals, excluding the original point fit.

| Outcome | Workers | Before | After | Time reduction |
|---|---:|---:|---:|---:|
| Continuous | 1 | 7.007 s | 4.417 s | 37.0% |
| Continuous | 4 | 2.694 s | 2.104 s | 21.9% |
| Binary | 1 | 3.142 s | 2.213 s | 29.6% |
| Binary | 4 | 1.243 s | 0.959 s | 22.9% |
| Cox | 1 | 2.984 s | 2.062 s | 30.9% |
| Cox | 4 | 1.086 s | 0.853 s | 21.5% |

All 18 paired comparisons produced byte-identical serialized fitted states,
including weights and stopping diagnostics. Predictions, bootstrap draws,
standard errors and interval endpoints matched exactly. The optimization
therefore leaves prediction and parameter errors unchanged on these runs.
This timing check uses one simulated dataset per outcome and four bootstrap
draws; it does not estimate statistical coverage or establish hardware-independent
speedups. It does not claim improved statistical accuracy.

The complete 41-test suite passed. The new regression checks all four
activations at widths 31, 32, 33 and 65, verifies prediction chunk boundaries,
and compares beta and network-weight gradients with independent scalar finite
differences. The compiler's vectorization report confirms vector instructions
for the packed accumulation loop. NHANES reporting tests also passed after
incorporating the concurrent curve-export update.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python reproduce/compare_dense_kernels.py --repeats 3 --bootstrap 4 \
  --jobs 1 4 --out results/dense_kernel_benchmark.json
```
