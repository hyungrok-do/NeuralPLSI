#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=eicu_mnar
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/simulation.log

module load singularity/3.9.8

singularity exec --nv --bind $SCRATCH --overlay $SCRATCH/containers/domain-generalization.ext3:ro \
        $SCRATCH/containers/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /ext3/miniconda3/bin/python simulation.py