#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu_medium
#SBATCH --mem=32GB
#SBATCH --job-name=NPLSI
#SBATCH --output=logs/simulation_%03a.log
#SBATCH -a 0-89
#SBATCH --exclude=cn-00[08-10]

source /etc/profile.d/modules.sh
module load singularity/3.9.8 2>/dev/null || module load singularity/3.9.8_rhel9 2>/dev/null || module load singularity/3.11.5

# Layout: 90 jobs total
#   NeuralPLSI: 2n × 3g × 3outcome × 4(ws×init) = 72 jobs  (IDs 0-71)
#   PLSI:       2n × 3g × 3outcome × 1(ws=0)     = 18 jobs  (IDs 72-89)

n_values=(200 1000)
g_values=(sigmoid sfun linear)
outcomes=(continuous binary cox)

ID=$SLURM_ARRAY_TASK_ID

if [ $ID -lt 72 ]; then
    # --- NeuralPLSI: 4-way ablation ---
    model="NeuralPLSI"
    outcome_idx=$(( ID % 3 ))
    g_idx=$(( (ID / 3) % 3 ))
    n_idx=$(( (ID / 9) % 2 ))
    combo_idx=$(( ID / 18 ))   # 0..3
    ws=$(( combo_idx % 2 ))
    init=$(( combo_idx / 2 ))

    n=${n_values[$n_idx]}
    g_fn=${g_values[$g_idx]}
    outcome=${outcomes[$outcome_idx]}

    echo "Running NeuralPLSI: n=$n, g_fn=$g_fn, outcome=$outcome, ws=$ws, init=$init"

    SCRATCH=/gpfs/scratch/doh03
    cd $SCRATCH/NeuralPLSI

    singularity exec --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
        $SCRATCH/containers/nvidia-cuda12.9.sif \
        /ext3/miniconda3/bin/python reproduce/main_simulation.py \
            --n_instances $n --g_fn $g_fn --outcome $outcome --models NeuralPLSI \
            --exposure_dist normal --warmstart $ws --initial $init
else
    # --- PLSI: ws=0 only ---
    model="PLSI"
    local_id=$(( ID - 72 ))
    outcome_idx=$(( local_id % 3 ))
    g_idx=$(( (local_id / 3) % 3 ))
    n_idx=$(( local_id / 9 ))

    n=${n_values[$n_idx]}
    g_fn=${g_values[$g_idx]}
    outcome=${outcomes[$outcome_idx]}

    echo "Running PLSI: n=$n, g_fn=$g_fn, outcome=$outcome, ws=0, init=0"

    SCRATCH=/gpfs/scratch/doh03
    cd $SCRATCH/NeuralPLSI

    singularity exec --bind $SCRATCH --overlay $SCRATCH/containers/survmix.ext3:ro \
        $SCRATCH/containers/nvidia-cuda12.9.sif \
        /ext3/miniconda3/bin/python reproduce/main_simulation.py \
            --n_instances $n --g_fn $g_fn --outcome $outcome --models PLSI \
            --exposure_dist normal --warmstart 0 --initial 0
fi
