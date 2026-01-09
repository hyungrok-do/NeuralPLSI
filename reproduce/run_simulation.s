#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-20:00:00
#SBATCH --partition=cpu_medium
#SBATCH --mem=32GB
#SBATCH --job-name=SIM
#SBATCH --output=logs/simulation_%03a.log
#SBATCH -a 0-53

module load singularity/3.9.8

n_values=(1000)
g_values=(sigmoid sfun linear)
outcomes=(continuous binary cox)
models=(NeuralPLSI PLSI)
exposure_dist=(normal uniform t)

outcome_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
g_idx=$(( (SLURM_ARRAY_TASK_ID / 3) % 3 ))
n_idx=0
model_idx=$(( (SLURM_ARRAY_TASK_ID / 9) % 2 ))
exposure_idx=$(( SLURM_ARRAY_TASK_ID / 18 ))

n=${n_values[$n_idx]}
g_fn=${g_values[$g_idx]}
outcome=${outcomes[$outcome_idx]}
model=${models[$model_idx]}
exposure_dist=${exposure_dist[$exposure_idx]}

echo "Running simulation with n=$n, g_fn=$g_fn, outcome=$outcome, model=$model, exposure_dist=$exposure_dist"

singularity exec --nv --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
    $SCRATCH/containers/nvidia-cuda12.9.sif \
    /ext3/miniconda3/bin/python reproduce/main_simulation.py --n_instances $n --g_fn $g_fn --outcome $outcome --model $model --exposure_dist $exposure_dist
