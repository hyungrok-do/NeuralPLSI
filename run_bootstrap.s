#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu_medium
#SBATCH --mem=32GB
#SBATCH --job-name=SIM
#SBATCH --output=logs/simulation_%02a.log
#SBATCH -a 0-35

module load singularity/3.9.8

# Define parameter arrays
n_values=(500 2000)
g_values=(linear sfun sigmoid)
outcomes=(continuous binary cox)
models=(NeuralPLSI PLSI)

# Total combinations = 2 * 3 * 3 * 2 = 36
# The array task ID will now range from 0 to 35.

# Compute the index for each parameter
n_idx=$(( SLURM_ARRAY_TASK_ID / 18 ))
g_idx=$(( (SLURM_ARRAY_TASK_ID % 18) / 6 ))
outcome_idx=$(( (SLURM_ARRAY_TASK_ID % 6) / 2 ))
model_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))

n=${n_values[$n_idx]}
g_fn=${g_values[$g_idx]}
outcome=${outcomes[$outcome_idx]}
model=${models[$model_idx]}

echo "Running simulation with n=$n, g_fn=$g_fn, outcome=$outcome, model=$model"

singularity exec --nv --bind $SCRATCH --overlay $SCRATCH/containers/domain-generalization.ext3:ro \
    $SCRATCH/containers/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /ext3/miniconda3/bin/python simulation_main.py --n_instances $n --g_fn $g_fn --outcome $outcome --model $model
