#!/bin/bash


# Get the directory of the current script
code_dir=${HOME}/local_scratch/code/pytorch_in_mri/papers/reconstruction/DIRECT
echo $code_dir
# Import functions from bash_functions.sh using a relative path
source "$code_dir/bash_functions.sh"

#
# Given a model name, relative to model_run/direct
# It evaluates on all possible anatomies and acceleration factors
# It stores it under paper/reconstruction/result
#
# The folder structure where the models are stored is given by
# model name
#   percentage
#     train dataset1
#       --> results
#     ...
#     train dataset3
#

username=$(whoami)


if [[ $username == "sharreve" ]]; then
    HOME=${HOME}"/local_scratch"
    MODEL_RUN="${HOME}/model_run/direct"
    # DINFERENCE="${HOME}/mri_data/cardiac_radial"
    DINFERENCE_RETRO="${HOME}/mri_data/cardiac_radial_inference_retrospective/input"
    # Not sure if this is the correct location \/
    DRESULT="${HOME}/paper/reconstruction/retro"
elif [[ $username == "20184098" ]]; then
    MODEL_RUN="${HOME}/model_run/direct"
    DINFERENCE_RETRO="${HOME}/data/direct_inference"
    DRESULT="${HOME}/data/paper/reconstruction/inference"
    module load cuda11.5/toolkit
    echo "Cuda visible devices " $CUDA_VISIBLE_DEVICES
else
    echo "Invalid username."
    exit 1
fi

percentage=false
overwrite=false
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --model_name|-m)
      model_name="$2"
      shift # Shift past the argument name
      shift # Shift past the argument value
      ;;
    --overwrite|-o)
      overwrite="$2"
      shift # Shift past the argument name
      shift # Shift past the argument value
      ;;
    --percentage|-p)
      percentage="$2"
      shift # Shift past the argument name
      shift # Shift past the argument value
      ;;
    --help|-h) # Handle --help and -h arguments
      display_help
      exit 0
      ;;
    *) # Handle unrecognized arguments
      echo "Unrecognized argument: $key"
      shift # Shift past the argument
      ;;
  esac
done

# Location of the DIRECT code
DIRECT_path="${HOME}/code/direct"

# Activate the direct environment
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate direct

# Move to directory of DIRECT
cd $DIRECT_path || exit
# Move to projects. That is where predict_val.py is stored
cd projects || exit

# model_name is for example 'kikinet_resume'
model_path="${MODEL_RUN}/${model_name}"

# Here we pick evaluation index 3
# This should be the validation dataset WITHOUT any masking (meaning ones)
validation_index=2
validation_str="undersampled"
echo "Acceleration factor " $validation_str

if [ "$percentage" == false ]; then
  percentage_list="${model_path}"/*/
else
  # TODO This is affected by sub-folder changes
  percentage_list="${model_path}/${percentage}p"
fi

# Using /*/ lists only dirs and no files
for train_percentage in $percentage_list; do
  # This refers to a specific model trained..
  train_percentage=$(basename "$train_percentage")
  config_file_path=$model_path"/config/inference_config_${train_percentage}_mixed.yaml"
  echo -e "\n\nTraining percentage \t" $train_percentage
  echo -e "Config file path    \t" $config_file_path
  if [[ "$train_percentage" != "config" ]]; then
    # Define the path that points to the percentage of the training data used
    percentage_model_path="${model_path}/${train_percentage}"
    # Loop over the models that were trained for the given training percentage
    for trained_model_path in "${percentage_model_path}"/*/; do
      trained_model_name=$(strip_dirname "$trained_model_path")
      echo -e "Trained model path      \t" $trained_model_path
      echo -e "Selected trained model  \t" $trained_model_name
      # Find the latest/newest model_*.pt file in the directory of the trained model
      checkpoint_path=$(find_model_file "${trained_model_path}")
      # checkpoint_path="${latest_file}"
      # Loop over the anatomies on which we want to evaluate
      #for anatomy in '2ch' 'transverse' 'sa' '4ch'; do
      # echo "Selected anatomy " $anatomy
      # Define the data for evaluation
      # temp_dinference="${DINFERENCE}/${anatomy}_split/test"
      temp_dinference="${DINFERENCE_RETRO}"
      # Define the result path
      # result_path="${DRESULT}/${model_name}/${train_percentage}/${trained_model_name}/${anatomy}/${validation_str}"
      result_path="${DRESULT}/${model_name}/${train_percentage}/${trained_model_name}/${validation_str}"
      # result_path="/local_scratch/sharreve/paper/reconstruction/inference/${model_name}/${train_percentage}/${trained_model_name}/${anatomy}/${validation_str}"
      # Create result path if it doesnt exist
      if [ ! -d "$result_path" ]; then
        mkdir -p "$result_path"
      else
        # If it already exists, and we DONT want to overwrite
        # Then continue..
        if [ "$overwrite" = false ]; then
          continue
        fi
      fi
      echo "============================="
      # Print the defined paths and directories
      echo "Data Path: $temp_dinference"
      echo "Checkpoint Path: $checkpoint_path"
      echo "Config file Path: $config_file_path"
      echo "Storage Path: $result_path"
      echo "============================="
      # Start the evaluation
      python3 predict_val.py $result_path --checkpoint $checkpoint_path --cfg $config_file_path --data-root $temp_dinference  --validation-index $validation_index
        # done
      done
    else
      echo "Provided percentage is not found. Please use 0, 25, 50, 75 or 100. Given: $train_percentage"
    fi
done
