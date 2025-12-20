# NeuralPLSI

Neural Partial-Linear Single Index (NeuralPLSI) is a method for modeling outcomes that depend on a linear combination of exposures (index) transformed by a flexible nonlinear function, adjusting for covariates.

Repository: [https://github.com/hyungrok-do/neuralplsi](https://github.com/hyungrok-do/neuralplsi)
Arxiv: [https://arxiv.org/abs/2512.11593](https://arxiv.org/abs/2512.11593)

## Overview

This package implements two models:
- **NeuralPLSI**: Uses a neural network to learn the nonlinear link function $g(\beta^T X)$.
- **PLSI**: Uses B-splines to estimate the link function (Wang et al., 2020).

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hyungrok-do/neuralplsi.git
cd neuralplsi
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Tutorial
See `tutorial.ipynb` for a step-by-step guide on generating data, fitting models, and visualizing results.

### 2. Simulation
The package includes a simulation suite to evaluate model performance.

```bash
# Run a simulation (e.g., n=500, sigmoid relationship, continuous outcome)
python reproduce/main_simulation.py --n_instances 500 --g_fn sigmoid --outcome continuous --models all

# Fully reproduce all simulation results
cd reproduce
./run_simulation.sh
cd ..
```

To visualize results:
```bash
python reproduce/visualize_simulation.py
```
This generates summary tables and plots in the `output/` directory.

### 3. NHANES Analysis
The NHANES dataset used in this example was downloaded from the [supplementary material](https://static-content.springer.com/esm/art%3A10.1186%2Fs12940-020-00644-4/MediaObjects/12940_2020_644_MOESM2_ESM.csv) of Wang et al. (2020).

To run the analysis (requires `NHANES/` directory):

```bash
python reproduce/nhanes.py
```
This script performs data preprocessing, model fitting, and bootstrap inference, saving results to `output/`.

## Key Components

- **`models/`**: Core implementation of `NeuralPLSI` and `SplinePLSI`.
- **`simulation/`**: Utilities for generating synthetic data.

## References

- Wang, Y., Wu, Y., Jacobson, M.H. et al. A family of partial-linear single-index models for analyzing complex environmental exposures with continuous, categorical, time-to-event, and longitudinal health outcomes. *Environ Health* **19**, 96 (2020). https://doi.org/10.1186/s12940-020-00644-4

## Cite As

```
@article{
  do2025neural,
  title={Neural Network-based Partial-Linear Single-Index Models for Environmental Mixtures Analysis},
  author={Do, Hyungrok and Wang, Yuyan and Liu, Mengling and Lee, Myeonggyun},
  journal={arXiv preprint arXiv:2512.11593},
  year={2025}
}
```
