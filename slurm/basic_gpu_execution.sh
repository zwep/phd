#!/bin/bash
#SBATCH --partition=bme.gpuresearch.q
#SBATCH --nodes=1
#SBATCH --nodelist=bme-gpuB001
#SBATCH --job-name=standard_60_minute_job
#SBATCH --time=00:10:0
#SBATCH --mem=1000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/bme001/20184098/data/report/output_%j.out

module load cuda11.6/toolkit
# Here we extract the FIRST argument. Which should be the script..
args=("$@")
script_path=("${args[@]:1}")
length_array=$((${#args[@]}-2))
remaining_args="${args[@]:2:$length_array}"

CODE_DIR="/home/bme001/20184098/code/pytorch_in_mri"
ENV_DIR="/home/bme001/20184098/code/pytorch_in_mri/venv/bin/activate"
full_path="$CODE_DIR/$script_path"

# Activate this environment so
source $ENV_DIR

# First find an available cuda decice..
CUDA_ID=$(python helper/nvidia_parser.py)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_ID: $CUDA_ID"
export CUDA_VISIBLE_DEVICES=$CUDA_ID
echo export CUDA_VISIBLE_DEVICES=$CUDA_ID
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "The following python environment is used"
echo "$(which python)"
echo "running the following file $full_path"
echo "$SLURM_JOB_ID" > "/home/bme001/20184098/latest_job_id.log"

srun python $full_path $remaining_args
