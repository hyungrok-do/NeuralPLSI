#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=PLSI
#SBATCH --output=logs/simulation_%02a.log
#SBATCH -a 0-23

module load singularity/3.9.8

# Define parameter arrays
n_values=(500 2000)
g_values=(linear logsquare sfun sigmoid)
outcomes=(continuous binary cox)

# Compute the index for each parameter
n_idx=$(( SLURM_ARRAY_TASK_ID / 12 ))
g_idx=$(( (SLURM_ARRAY_TASK_ID % 12) / 3 ))
outcome_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))

n=${n_values[$n_idx]}
g_fn=${g_values[$g_idx]}
outcome=${outcomes[$outcome_idx]}

echo "Running simulation with n=$n, g_fn=$g_fn, outcome=$outcome"

singularity exec --nv --bind $SCRATCH --overlay $SCRATCH/containers/domain-generalization.ext3:ro \
    $SCRATCH/containers/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /ext3/miniconda3/bin/python test.py --n $n --g_fn $g_fn --outcome $outcome
