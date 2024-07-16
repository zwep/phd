#!/bin/bash
#SBATCH --partition=bme.gpuresearch.q
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --job-name=standard_job
#SBATCH --time=04:20:0
#SBATCH --mem=4000M
#SBATCH --output=/home/bme001/20184098/data/report/output_%j.out

args=("$@")
echo "Args received: $@"
PYTHON_SCRIPT_PATH=("${args[@]:1}")
length_array=$((${#args[@]}-2))
remaining_args="${args[@]:2:$length_array}"

# Create the full python path using my home dir
CODE_DIR="/home/bme001/20184098/code/pytorch_in_mri"
ENV_DIR="/home/bme001/20184098/code/pytorch_in_mri/venv/bin/activate"
FULL_PYTHON_PATH="$CODE_DIR/$PYTHON_SCRIPT_PATH"

# Activate this environment so
source $ENV_DIR
echo "Following script is run $FULL_PYTHON_PATH"
echo "With remaining args $remaining_args"
echo "$SLURM_JOB_ID" > "/home/bme001/20184098/latest_job_id.log"
srun python $FULL_PYTHON_PATH $remaining_args
