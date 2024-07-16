#!/bin/bash
#SBATCH --partition=bme.gpuresearch.q
#SBATCH --nodes=1
#SBATCH --job-name=nnunet_run
#SBATCH --time=128:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10000M
#SBATCH --output=/home/bme001/20184098/data/report/output_%j.out
#SBATCH --nodelist=bme-gpuA001

module load cuda11.6/toolkit
# Here we extract the FIRST argument. Which should be the task number
args=("$@")
task_number=("${args[@]:1}")
echo "Found the following task $task_number"
ENV_DIR="/home/bme001/20184098/code/pytorch_in_mri/venv/bin/activate"
# Activate this environment so
source $ENV_DIR
srun nnUNet_train 2d nnUNetTrainerV2 "$task_number" all
