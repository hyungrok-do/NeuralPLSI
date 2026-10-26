#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=sandwich
#SBATCH --output=logs/sandwich_%A_%a.log
#SBATCH --array=0-5
# The sandwich SE and a B = 500 cold-start bootstrap of the same L-BFGS fits,
# 200 datasets per setting. Array IDs 0-5: continuous then binary, linear,
# sigmoid, threshold. Submit from the repository root; rows go to output/sandwich/matched.
set -euo pipefail
# The container starts in the home directory unless the working directory is bound and set.
cd "${SLURM_SUBMIT_DIR}"
id=${SLURM_ARRAY_TASK_ID}
outcomes=(continuous binary); links=(linear sigmoid threshold)
outcome=${outcomes[$((id / 3))]}; g=${links[$((id % 3))]}
SIF=/gpfs/scratch/doh03/containers/nvidia-cuda12.9.sif
OVERLAY=/gpfs/scratch/doh03/containers/survmix.ext3
SING=/gpfs/share/apps/singularity/3.9.8/bin/singularity
export NEURALPLSI_CACHE_DIR=$PWD/.cache/sandwich_$id
mkdir -p "$NEURALPLSI_CACHE_DIR" output/sandwich/matched logs
echo "task $id: $outcome $g on $(hostname), $(date)"
"$SING" exec --bind "$PWD" --pwd "$PWD" --overlay "$OVERLAY:ro" --env "CXX=g++,NEURALPLSI_CACHE_DIR=$NEURALPLSI_CACHE_DIR,PYTHONWARNINGS=ignore,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
  "$SIF" /ext3/miniconda3/bin/python reproduce/sandwich/study.py "$outcome" "$g" 0 200 500 \
  "output/sandwich/matched/${outcome}_${g}.jsonl"
echo "done $(date)"
