#!/bin/bash
#SBATCH --job-name=neuralplsi_ablation
#SBATCH --output=ablation_grid_%j.out
#SBATCH --error=ablation_grid_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Note: Adjust the partition, memory, and cpvs-per-task to match your HPC environment.

echo "Starting NeuralPLSI Ablation Grid Generation"
date

# Load necessary modules (adjust based on HPC environment)
module load python/3.10 || module load miniconda3/4.6.14 || true
pip install --user joblib pandas lifelines scikit-learn

# Execute the python driver script
python reproduce/run_ablation_grid.py

echo "Ablation Grid complete."
date
