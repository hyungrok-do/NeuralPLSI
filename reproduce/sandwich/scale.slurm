#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=scale
#SBATCH --output=logs/scale_%A_%a.log
#SBATCH --array=0-53
# AdamW + frozen correction at large n and wide networks (scale_monte_carlo.py):
# 200 datasets per cell, one cell per array ID: n in (2000, 20000, 100000) x
# width in (32, 128, 512) x outcome in (continuous, binary, cox) x g in
# (sigmoid, threshold). Submit from the repository root; rows go to output/sandwich/scale.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
id=${SLURM_ARRAY_TASK_ID}
ns=(2000 20000 100000); widths=(32 128 512); outcomes=(continuous binary cox); links=(sigmoid threshold)
n=${ns[$((id / 18))]}; width=${widths[$(((id / 6) % 3))]}; outcome=${outcomes[$(((id / 2) % 3))]}; g=${links[$((id % 2))]}
SIF=/gpfs/scratch/doh03/containers/nvidia-cuda12.9.sif
OVERLAY=/gpfs/scratch/doh03/containers/survmix.ext3
SING=/gpfs/share/apps/singularity/3.9.8/bin/singularity
export NEURALPLSI_CACHE_DIR=$PWD/.cache/scale
mkdir -p "$NEURALPLSI_CACHE_DIR" output/sandwich/scale logs
echo "task $id: n=$n width=$width $outcome $g on $(hostname), $(date)"
"$SING" exec --bind "$PWD" --pwd "$PWD" --overlay "$OVERLAY:ro" \
  --env "CXX=g++,NEURALPLSI_CACHE_DIR=$NEURALPLSI_CACHE_DIR,PYTHONWARNINGS=ignore,OMP_NUM_THREADS=1,SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" \
  "$SIF" /ext3/miniconda3/bin/python reproduce/sandwich/scale_monte_carlo.py "$n" "$width" "$outcome" "$g" 0 200 \
  "output/sandwich/scale/${n}_${width}_${outcome}_${g}.jsonl"
echo "done $(date)"
