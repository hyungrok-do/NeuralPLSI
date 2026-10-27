# Scalability: fit time, accuracy and sandwich time as n grows

One fit per dataset on one thread, for NeuralPLSI fitted by L-BFGS (the package
default), by minibatch AdamW + frozen correction and by minibatch AdamW + exact
gamma (the comparator, "no correction" of beta), and for SplinePLSI. Sigmoid g,
8 exposures and 3 confounders, a 2 x 32 network, n from 500 to 100,000 training
rows, 3 datasets per n (seeds 0-2, fit seed 1337). The engine's fits are
single-threaded (only its bootstrap and the full sandwich's Hessian use
threads), and the fits ran one at a time on an Apple-silicon laptop, so times
compare across methods. Accuracy is the angle between the fitted and the true
beta. The sandwich is the full one for L-BFGS (up to n = 20,000) and the frozen
one for AdamW + frozen correction, both on one thread; the benchmark times it
for continuous and binary outcomes. Medians over the 3 datasets;
`scalability.csv` has every n (500, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000,
100,000).

Columns: L-BFGS | AdamW + frozen correction | AdamW + exact gamma | SplinePLSI;
sandwich: L-BFGS full | AdamW + frozen correction frozen.

| n | L-BFGS | AdamW + frozen | AdamW + exact γ | SplinePLSI | L-BFGS | AdamW + frozen | AdamW + exact γ | SplinePLSI | full | frozen |
|---|---|---|---|---|---|---|---|---|---|---|
| | **continuous: fit (s)** | | | | **angle (°)** | | | | **sandwich (s)** | |
| 500 | 0.13 | 0.06 | 0.05 | 0.01 | 6.12 | 7.09 | 9.20 | 6.65 | 1.31 | 0.001 |
| 2,000 | 0.25 | 0.12 | 0.11 | 0.07 | 2.94 | 2.97 | 3.00 | 3.06 | 4.82 | 0.002 |
| 10,000 | 1.13 | 0.24 | 0.17 | 0.30 | 0.82 | 0.69 | 1.18 | 0.76 | 23.2 | 0.008 |
| 20,000 | 2.38 | 0.37 | 0.24 | 0.64 | 0.68 | 0.69 | 1.50 | 0.69 | 46.6 | 0.016 |
| 100,000 | 13.2 | 2.19 | 1.18 | 3.06 | 0.31 | 0.31 | 1.06 | 0.32 |  | 0.081 |
| | **binary: fit (s)** | | | | **angle (°)** | | | | **sandwich (s)** | |
| 500 | 0.07 | 0.05 | 0.04 | 0.04 | 13.0 | 12.2 | 18.5 | 13.5 | 1.34 | 0.001 |
| 2,000 | 0.22 | 0.11 | 0.10 | 0.16 | 4.09 | 4.29 | 4.56 | 3.93 | 4.85 | 0.002 |
| 10,000 | 1.12 | 0.31 | 0.25 | 0.67 | 3.03 | 3.03 | 3.15 | 3.16 | 23.2 | 0.008 |
| 20,000 | 2.41 | 0.47 | 0.38 | 1.16 | 1.54 | 1.49 | 1.94 | 1.46 | 46.8 | 0.017 |
| 100,000 | 9.82 | 2.57 | 1.61 | 6.21 | 0.57 | 0.55 | 1.08 | 0.62 |  | 0.083 |
| | **cox: fit (s)** | | | | **angle (°)** | | | | **sandwich (s)** | |
| 500 | 0.06 | 0.07 | 0.06 | 0.04 | 4.91 | 4.88 | 5.25 | 5.13 |  |  |
| 2,000 | 0.20 | 0.13 | 0.12 | 0.21 | 3.37 | 3.49 | 3.49 | 3.42 |  |  |
| 10,000 | 0.90 | 0.22 | 0.16 | 0.82 | 1.15 | 1.16 | 1.23 | 1.06 |  |  |
| 20,000 | 1.81 | 0.44 | 0.31 | 1.69 | 0.93 | 0.95 | 1.50 | 0.97 |  |  |
| 100,000 | 9.13 | 1.65 | 1.05 | 7.95 | 0.47 | 0.48 | 1.69 | 0.45 |  |  |

- **All fit in time about linear in n.** At n = 100,000 L-BFGS takes 9-13 s,
  AdamW + frozen correction 1.7-2.6 s, SplinePLSI 3-8 s.
- **The frozen correction removes AdamW's accuracy plateau.** Early stopping
  ends AdamW's training after fewer epochs as n grows (16 at n = 100,000, Cox);
  with only gamma refitted, beta stays 1.1-1.9 degrees from the truth from n =
  20,000 on. With the frozen correction it tracks L-BFGS at every n and outcome
  (within the noise of 3 datasets; 0.31-0.55 degrees at n = 100,000), at 4-6
  times less time.
- **The frozen sandwich costs almost nothing**: 0.08 s at n = 100,000, against
  47 s for the full sandwich at n = 20,000 (one thread; it needs a dense d x d
  Hessian, d = 1,164 here, and is limited to 8,000 weights). A bootstrap of B
  refits costs about B fits divided by the threads.

## Reproduce

```bash
python reproduce/scalability/benchmark.py        # about an hour; rows in output/scalability/fits.jsonl
python reproduce/scalability/summarize.py
```
