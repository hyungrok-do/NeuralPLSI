#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --mem=32GB
#SBATCH --job-name=NPLSI
#SBATCH --output=logs/simulation_%03a.log
#SBATCH -a 0-53
#SBATCH --exclude=cn-00[08-10]

source /etc/profile.d/modules.sh
module load singularity/3.9.8 2>/dev/null || module load singularity/3.9.8_rhel9 2>/dev/null || module load singularity/3.11.5

# Layout: 54 jobs total
#   NeuralPLSI: 2n × 3g × 3outcome × 2(init) = 36 jobs  (IDs 0-35)
#   PLSI:       2n × 3g × 3outcome           = 18 jobs  (IDs 36-53)

n_values=(200 1000)
g_values=(sigmoid sfun linear)
outcomes=(continuous binary cox)

ID=$SLURM_ARRAY_TASK_ID

if [ $ID -lt 36 ]; then
    # --- NeuralPLSI: Naive vs GLM init ---
    model="NeuralPLSI"
    outcome_idx=$(( ID % 3 ))
    g_idx=$(( (ID / 3) % 3 ))
    n_idx=$(( (ID / 9) % 2 ))
    init=$(( ID / 18 ))   # 0..1

    n=${n_values[$n_idx]}
    g_fn=${g_values[$g_idx]}
    outcome=${outcomes[$outcome_idx]}

    echo "Running NeuralPLSI: n=$n, g_fn=$g_fn, outcome=$outcome, init=$init"

    SCRATCH=/gpfs/scratch/doh03
    cd $SCRATCH/NeuralPLSI

    singularity exec --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
        $SCRATCH/containers/nvidia-cuda12.9.sif \
        /ext3/miniconda3/bin/python reproduce/main_simulation.py \
            --n_instances $n --g_fn $g_fn --outcome $outcome --models NeuralPLSI \
            --exposure_dist normal --initial $init --n_bootstrap 100
else
    # --- PLSI ---
    model="PLSI"
    local_id=$(( ID - 36 ))
    outcome_idx=$(( local_id % 3 ))
    g_idx=$(( (local_id / 3) % 3 ))
    n_idx=$(( local_id / 9 ))

    n=${n_values[$n_idx]}
    g_fn=${g_values[$g_idx]}
    outcome=${outcomes[$outcome_idx]}

    echo "Running PLSI: n=$n, g_fn=$g_fn, outcome=$outcome, init=0"

    SCRATCH=/gpfs/scratch/doh03
    cd $SCRATCH/NeuralPLSI

    singularity exec --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
        $SCRATCH/containers/nvidia-cuda12.9.sif \
        /ext3/miniconda3/bin/python reproduce/main_simulation.py \
            --n_instances $n --g_fn $g_fn --outcome $outcome --models PLSI \
            --exposure_dist normal --initial 0 --n_bootstrap 100
fi
