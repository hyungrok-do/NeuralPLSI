#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu_medium
#SBATCH --mem=16GB
#SBATCH --job-name=NPLSI_abl
#SBATCH --output=logs/ablation_%03a.log
#SBATCH -a 0-8
#SBATCH --exclude=cn-00[08-10]

source /etc/profile.d/modules.sh
module load singularity/3.9.8 2>/dev/null || module load singularity/3.9.8_rhel9 2>/dev/null || module load singularity/3.11.5

outcomes=(continuous binary cox)
g_values=(linear sigmoid sfun)

outcome_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
g_idx=$(( SLURM_ARRAY_TASK_ID / 3 ))

outcome=${outcomes[$outcome_idx]}
g_fn=${g_values[$g_idx]}

echo "Running ablation: outcome=$outcome, g_fn=$g_fn (4-way NeuralPLSI + PLSI ws=0)"

SCRATCH=/gpfs/scratch/doh03
cd $SCRATCH/NeuralPLSI

singularity exec --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
    $SCRATCH/containers/nvidia-cuda12.9.sif \
    /ext3/miniconda3/bin/python reproduce/ablation_warmstart.py \
        --outcome $outcome --g_fn $g_fn --n 500 --n_reps 20
