#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --partition=cpu_short
#SBATCH --job-name=PLSI
#SBATCH --output=logs/simulation_%02a.log
#SBATCH -a 0-17

module load singularity/3.9.8

# Define parameter arrays
n_values=(500 2000)
g_values=(linear sfun sigmoid)  # Removed logsquare
outcomes=(continuous binary cox)

# Total combinations = 2 * 3 * 3 = 18
# Compute the index for each parameter
n_idx=$(( SLURM_ARRAY_TASK_ID / 9 ))
g_idx=$(( (SLURM_ARRAY_TASK_ID % 9) / 3 ))
outcome_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))

n=${n_values[$n_idx]}
g_fn=${g_values[$g_idx]}
outcome=${outcomes[$outcome_idx]}

echo "Running simulation with n=$n, g_fn=$g_fn, outcome=$outcome"

singularity exec --nv --bind $SCRATCH --overlay $SCRATCH/containers/domain-generalization.ext3:ro \
    $SCRATCH/containers/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /ext3/miniconda3/bin/python test.py --n $n --g_fn $g_fn --outcome $outcome
