# Scalability: fit time and accuracy as n grows

One fit per dataset on one thread, for L-BFGS NeuralPLSI (the package default),
minibatch AdamW NeuralPLSI with early stopping (the baseline of the main
simulation) and SplinePLSI. Sigmoid g, 8 exposures and 3 confounders, n from
500 to 100,000 training rows, 3 datasets per n (seeds 0-2, fit seed 1337). The
engine's fits are single-threaded (only its bootstrap uses threads), and the
fits ran one at a time on an Apple-silicon laptop, so times compare across
methods. Accuracy is the angle between the fitted and the true β. Medians over
the 3 datasets; `scalability.csv` has every cell.

| n | L-BFGS NN | AdamW NN | Spline | L-BFGS NN | AdamW NN | Spline | NN sandwich |
|---|---|---|---|---|---|---|---|
| | **fit (s), continuous** | | | **angle (°), continuous** | | | **(s)** |
| 500 | 0.11 | 0.05 | 0.01 | 6.1 | 9.2 | 6.7 | 0.8 |
| 2,000 | 0.23 | 0.11 | 0.07 | 2.9 | 3.0 | 3.1 | 2.3 |
| 10,000 | 1.1 | 0.18 | 0.33 | 0.82 | 1.18 | 0.76 | 11.1 |
| 20,000 | 2.4 | 0.24 | 0.61 | 0.68 | 1.50 | 0.69 | 23.1 |
| 100,000 | 13.6 | 1.2 | 3.1 | 0.31 | 1.06 | 0.32 | |
| | **fit (s), binary** | | | **angle (°), binary** | | | |
| 500 | 0.07 | 0.04 | 0.04 | 13.0 | 18.5 | 13.5 | 0.7 |
| 2,000 | 0.24 | 0.10 | 0.16 | 4.1 | 4.6 | 3.9 | 2.3 |
| 10,000 | 1.1 | 0.25 | 0.68 | 3.0 | 3.2 | 3.2 | 11.2 |
| 20,000 | 2.4 | 0.37 | 1.2 | 1.54 | 1.94 | 1.46 | 23.2 |
| 100,000 | 10.1 | 1.6 | 6.3 | 0.57 | 1.08 | 0.62 | |
| | **fit (s), Cox** | | | **angle (°), Cox** | | | |
| 500 | 0.06 | 0.06 | 0.04 | 4.9 | 5.3 | 5.1 | |
| 2,000 | 0.20 | 0.12 | 0.21 | 3.4 | 3.5 | 3.4 | |
| 10,000 | 0.90 | 0.16 | 0.83 | 1.15 | 1.23 | 1.06 | |
| 20,000 | 1.8 | 0.30 | 1.7 | 0.93 | 1.50 | 0.97 | |
| 100,000 | 9.4 | 1.0 | 8.1 | 0.47 | 1.69 | 0.45 | |

- **All three fit in time about linear in n.** L-BFGS NeuralPLSI takes about
  0.10-0.14 ms per row (10-14 s at n = 100,000), SplinePLSI 0.03-0.08 ms per row.
  For Cox outcomes the two take about the same time.
- **AdamW is the fastest and improves the least with n.** Early stopping ends its
  training after fewer epochs as n grows (16 at n = 100,000, Cox). From
  n = 20,000 to 100,000 its angle to the true β stays at 1.1-1.7° for continuous
  and Cox outcomes and falls only from 1.9° to 1.1° for binary, while L-BFGS
  NeuralPLSI and SplinePLSI reach 0.3-0.6° at n = 100,000 (2-4 times smaller). At
  n = 500, binary, it is also the least accurate (18.5° against 13.0°).
- **L-BFGS NeuralPLSI and SplinePLSI are equally accurate** at every n for sigmoid
  g, where the spline's 5 knots suffice.
- **Inference cost.** The full-parameter sandwich of an L-BFGS fit
  (`results/sandwich`) takes about 1.2 ms per row on one thread, about 10 fits'
  worth; a bootstrap with B refits costs B fits divided by the threads (at
  n = 20,000, B = 1,000: about 2,400 CPU-seconds, against 23 s for the sandwich).
  These are fits with the package's stopping rule; converged fits, which the
  sandwich needs, take about 2-5 times longer.

## Reproduce

```bash
python reproduce/scalability/benchmark.py        # about 30 minutes; rows in output/scalability/fits.jsonl
python reproduce/scalability/summarize.py
```
