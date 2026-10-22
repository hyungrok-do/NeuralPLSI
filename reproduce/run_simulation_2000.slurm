#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --mem=32GB
#SBATCH --job-name=NPLSI_2000
#SBATCH --output=logs/simulation_2000_%A_%a.log
#SBATCH --array=0-8
#SBATCH --exclude=cn-00[08-10]

# IDs 0-8: NeuralPLSI. Submit --array=0-17 to include SplinePLSI, or 18-26 for
# the minibatch AdamW baseline.
set -euo pipefail
cd "${NPLSI_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
bash reproduce/run_simulation_hpc.sh 2000
