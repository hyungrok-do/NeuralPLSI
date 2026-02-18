#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu_medium
#SBATCH --mem=32GB
#SBATCH --job-name=NPLSI
#SBATCH --output=logs/simulation_%03a.log
#SBATCH -a 0-71
#SBATCH --exclude=cn-00[08-10]

source /etc/profile.d/modules.sh
module load singularity/3.9.8 2>/dev/null || module load singularity/3.9.8_rhel9 2>/dev/null || module load singularity/3.11.5

n_values=(200 1000)
g_values=(sigmoid sfun linear)
outcomes=(continuous binary cox)
models=(NeuralPLSI PLSI)
warmstart_values=(0 1)

# Total combos: 2 n × 3 g × 3 outcome × 2 model × 2 warmstart = 72
outcome_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
g_idx=$(( (SLURM_ARRAY_TASK_ID / 3) % 3 ))
model_idx=$(( (SLURM_ARRAY_TASK_ID / 9) % 2 ))
n_idx=$(( (SLURM_ARRAY_TASK_ID / 18) % 2 ))
ws_idx=$(( SLURM_ARRAY_TASK_ID / 36 ))

n=${n_values[$n_idx]}
g_fn=${g_values[$g_idx]}
outcome=${outcomes[$outcome_idx]}
model=${models[$model_idx]}
ws=${warmstart_values[$ws_idx]}

echo "Running simulation with n=$n, g_fn=$g_fn, outcome=$outcome, model=$model, warmstart=$ws"

SCRATCH=/gpfs/scratch/doh03
cd $SCRATCH/NeuralPLSI

singularity exec --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
    $SCRATCH/containers/nvidia-cuda12.9.sif \
    /ext3/miniconda3/bin/python reproduce/main_simulation.py \
        --n_instances $n --g_fn $g_fn --outcome $outcome --models $model \
        --exposure_dist normal --warmstart $ws
