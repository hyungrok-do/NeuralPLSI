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

# Load Singularity module
source /etc/profile.d/modules.sh
module load singularity/3.9.8 2>/dev/null || module load singularity/3.9.8_rhel9 2>/dev/null || module load singularity/3.11.5

SCRATCH=/gpfs/scratch/doh03
cd $SCRATCH/NeuralPLSI

# Execute the python driver script within the container
singularity exec --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
    $SCRATCH/containers/nvidia-cuda12.9.sif \
    /ext3/miniconda3/bin/python reproduce/run_ablation_grid.py

echo "Ablation Grid complete."
date
