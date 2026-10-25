#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --job-name=sandwich
#SBATCH --output=logs/sandwich_%A_%a.log
#SBATCH --array=0-11
# The sandwich SE and a B = 500 cold-start bootstrap of the same fits, 200
# datasets per setting. Array IDs 0-5: converged fits (build_converged.sh); 6-11:
# the package stopping rule. Within each: continuous then binary, linear,
# sigmoid, threshold. Submit from the repository root; rows go to output/sandwich/matched.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
id=${SLURM_ARRAY_TASK_ID}
arms=(converged current); outcomes=(continuous binary); links=(linear sigmoid threshold)
arm=${arms[$((id / 6))]}; outcome=${outcomes[$(((id % 6) / 3))]}; g=${links[$((id % 3))]}
SIF=/gpfs/scratch/doh03/containers/nvidia-cuda12.9.sif
OVERLAY=/gpfs/scratch/doh03/containers/survmix.ext3
SING=/gpfs/share/apps/singularity/3.9.8/bin/singularity
export NEURALPLSI_CACHE_DIR=$PWD/.cache/sandwich_$id
mkdir -p "$NEURALPLSI_CACHE_DIR" output/sandwich/matched logs
lib=
if [ "$arm" = converged ]; then
  lib=$NEURALPLSI_CACHE_DIR/libneuralplsi_converged.so
  "$SING" exec --overlay "$OVERLAY:ro" --env CXX=g++ "$SIF" sh reproduce/sandwich/build_converged.sh "$lib"
fi
echo "task $id: $arm $outcome $g on $(hostname), $(date)"
"$SING" exec --overlay "$OVERLAY:ro" --env "NEURALPLSI_LIBRARY=$lib,NEURALPLSI_CACHE_DIR=$NEURALPLSI_CACHE_DIR,PYTHONWARNINGS=ignore,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
  "$SIF" /ext3/miniconda3/bin/python reproduce/sandwich/study.py "$arm" "$outcome" "$g" 0 200 500 \
  "output/sandwich/matched/${arm}_${outcome}_${g}.jsonl"
echo "done $(date)"
