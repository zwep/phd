#!/usr/bin/bash

model_name=unet_RADIAL
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --model_name|-m)
      model_name="$2"
      shift # Shift past the argument name
      shift # Shift past the argument value
      ;;
    *) # Handle unrecognized arguments
      echo "Unrecognized argument: $key"
      shift # Shift past the argument
      ;;
  esac
done


model_base_name=$(echo "$model_name" | awk -F'_' '{output=""; for(i=1; i<=NF-1; i++) output=output $i; print output}')
type_name=$(echo "$model_name" | awk -F'_' '{print $NF}')

# Activate the direct environment`
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate /local_scratch/sharreve/anaconda3/envs/venv

cd ${HOME}/local_scratch/code/pytorch_in_mri

papers/reconstruction/DIRECT/evaluate_model.sh -m $model_name -o
# python data_postproc/objective/reconstruction/visualize_results_model.py -model $model_name -filter '/mixed/'
python data_postproc/objective/reconstruction/calculate_metrics_model.py -model $model_name -filter '/mixed/'
# python data_postproc/objective/reconstruction/visualize_metrics_model.py -model $model_name

# Visualize range of percentages
# python data_postproc/objective/reconstruction/visualize_results_per_percentage.py -model $model_name -acc 5
# python data_postproc/objective/reconstruction/visualize_results_per_percentage.py -model $model_name -acc 10
