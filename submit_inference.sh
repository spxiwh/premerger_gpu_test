#!/bin/bash
#
#SBATCH --job-name=pre-merger
#SBATCH --time=1-00:00
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu[03,04,05,06,07]
#SBATCH --output=logs/pre-merger-%j.out
#SBATCH --error=logs/pre-merger-%j.err

module purge
module load system
module add anaconda3

CONDA_ENV_NAME=lisa-gs-premerger

echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Using conda environment: $CONDA_ENV_NAME"

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

echo "Running inference with config.yaml"
conda run -n $CONDA_ENV_NAME python run_inference.py config.yaml
echo "Inference completed at $(date)"