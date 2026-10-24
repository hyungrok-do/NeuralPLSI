#!/usr/bin/env bash
# Shared array-job entry point. Run from the repository root.
set -euo pipefail
n=${1:?Pass the sample size}
id=${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID (0-35)}
[[ "$id" =~ ^[0-9]+$ ]] && (( id <= 35 )) || { echo 'Array ID must be 0-35' >&2; exit 2; }
g_values=(sigmoid threshold linear)
outcomes=(continuous binary cox)
# 0-8 NeuralPLSI, 9-17 SplinePLSI, 18-26 the minibatch AdamW baseline,
# 27-35 the baseline followed by the exact gamma fit.
model=NeuralPLSI
(( id < 9 )) || model=PLSI
(( id < 18 )) || model=NeuralPLSI-Adam
(( id < 27 )) || model=NeuralPLSI-AdamGamma
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
project=${NPLSI_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$project"
backend=${NN_BACKEND:-cpp}
out=${OUT_DIR:-output/simulation}
args=(reproduce/main_simulation.py --n_instances "$n"
      --g_fn "${g_values[$(( (id / 3) % 3 ))]}" --outcome "${outcomes[$(( id % 3 ))]}"
      --models "$model" --exposure_dist normal --n_bootstrap "${N_BOOTSTRAP:-100}"
      --n_replicates "${N_REPLICATES:-500}" --seed0 "${SEED0:-0}"
      --bootstrap_jobs "${BOOTSTRAP_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
      --device cpu --nn_backend "$backend" --out "$out" --save_every 1)
[[ ${SAVE_BOOTSTRAP_SAMPLES:-0} == 1 ]] && args+=(--save_bootstrap_samples)
[[ ${NPLSI_INITIAL:-0} == 1 ]] && args+=(--initial)
# Set NPLSI_CONTAINER for Singularity, or omit it to use the active Python env.
if [[ -n ${NPLSI_CONTAINER:-} ]]; then
    options=(exec --bind "$project")
    [[ -n ${NPLSI_OVERLAY:-} ]] && options+=(--overlay "${NPLSI_OVERLAY}:ro")
    run=(singularity "${options[@]}" "$NPLSI_CONTAINER" "${NPLSI_PYTHON:-/ext3/miniconda3/bin/python}" "${args[@]}")
else
    run=("${NPLSI_PYTHON:-python}" "${args[@]}")
fi
if [[ ${NPLSI_DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${run[@]}"
    printf '\n'
    exit 0
fi
if [[ -n ${NPLSI_CONTAINER:-} ]] && ! command -v singularity >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
    module load singularity/3.9.8 2>/dev/null || module load singularity/3.9.8_rhel9 2>/dev/null || module load singularity/3.11.5
fi
exec "${run[@]}"
