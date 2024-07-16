#!/bin/bash
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --job-name=standard_10_minute_job
#SBATCH --time=00:30:0
#SBATCH --output=/home/bme001/20184098/data/report/output_%j.out

# This was usefull to process the number of arguments..
# But it did not allow for arguments to pass on to the python script
#while [[ "$#" -gt 0 ]]; do
#    case $1 in
#        -s|--script) script_path="$2"; shift ;;
#        *) echo "Unknown parameter passed: $1"; exit 1 ;;
#    esac
#    shift
#done

# We have solved that with ChatGPT and this piece of code
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
srun python $full_path $remaining_args
