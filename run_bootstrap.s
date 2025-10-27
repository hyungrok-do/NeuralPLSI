#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-20:00:00
#SBATCH --partition=cpu_medium
#SBATCH --mem=32GB
#SBATCH --job-name=SIM
#SBATCH --output=logs/simulation_%02a.log
#SBATCH -a 0-53

module load singularity/3.9.8

# Define parameter arrays
n_values=(1000)
g_values=(sigmoid sfun linear)
outcomes=(continuous binary cox)
models=(NeuralPLSI PLSI)
exposure_dist=(uniform normal t)

# Total combinations = 1 * 3 * 3 * 2 * 3 = 54
# The array task ID will now range from 0 to 53.

# Compute the index for each parameter
# Order: n_values(1) * g_values(3) * outcomes(3) * models(2) * exposure_dist(3)
# So we have: exposure_dist varies fastest, then model, then outcome, then g_fn, then n
exposure_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
model_idx=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
outcome_idx=$(( (SLURM_ARRAY_TASK_ID / 6) % 3 ))
g_idx=$(( (SLURM_ARRAY_TASK_ID / 18) % 3 ))
n_idx=$(( SLURM_ARRAY_TASK_ID / 54 ))

n=${n_values[$n_idx]}
g_fn=${g_values[$g_idx]}
outcome=${outcomes[$outcome_idx]}
model=${models[$model_idx]}
exposure_dist=${exposure_dist[$exposure_idx]}

echo "Running simulation with n=$n, g_fn=$g_fn, outcome=$outcome, model=$model, exposure_dist=$exposure_dist"

singularity exec --nv --bind $SCRATCH --overlay $SCRATCH/containers/domain-generalization.ext3:ro \
    $SCRATCH/containers/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /ext3/miniconda3/bin/python simulation_main.py --n_instances $n --g_fn $g_fn --outcome $outcome --model $model --exposure_dist $exposure_dist
