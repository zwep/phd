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
    # Location where the trained models are
    HOME=${HOME}"/local_scratch"
    DDATA="${HOME}/mri_data/cardiac_full_radial"
    # Location of results
    DRESULT="${HOME}/paper/reconstruction/results"
elif [[ $username == "20184098" ]]; then
    DDATA="${HOME}/data/direct"
    module load cuda11.5/toolkit
    echo "Cuda visible devices " $CUDA_VISIBLE_DEVICES
    DRESULT="${HOME}/data/paper/reconstruction/results"
else
    echo "Invalid username."
    exit 1
fi

# Location of trained models
TRAINED_path="${HOME}/model_run/direct"
# Location of the DIRECT code
DIRECT_path="${HOME}/code/direct"




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

# Activate the direct environment
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate direct

# Move to directory of DIRECT
cd $DIRECT_path || exit
# Move to projects. That is where predict_val.py is stored
cd projects || exit

# model_name is for example 'kikinet_resume'
model_path="${TRAINED_path}/${model_name}"

if [ "$percentage" == false ]; then
  percentage_list="${model_path}"/*/
else
  # TODO This is affected by sub-folder changes
  percentage_list="${model_path}/${percentage}p"
fi

echo "debug " $percentage_list
# Using /*/ lists only dirs and no files
for train_percentage in $percentage_list; do
  # This refers to a specific model trained..
  train_percentage=$(basename "$train_percentage")
      echo -e "\n\nTraining percentage \t" $train_percentage
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

      experiment_dir=$(strip_filename "$checkpoint_path")
      # Loop over the anatomies on which we want to evaluate
      # for anatomy in '2ch' 'transverse' 'sa' '4ch'; do
      for anatomy in 'mixed'; do
        echo "Selected anatomy " $anatomy
        # Define the data for evaluation
        # HARDCODED
        data_path="${DDATA}/${anatomy}/test/input"
        # Loop over the undersampling factors on which we want to evaluate
        for validation_index in 0 1; do
          if [[ $validation_index -eq 0 ]]; then
            validation_str="5x"
          else
            validation_str="10x"
          fi
          echo "Acceleration factor " $validation_str
          # Define the result path
          result_path="${DRESULT}/${model_name}/${train_percentage}/${trained_model_name}/${anatomy}/${validation_str}"
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
          echo "Data Path: $data_path"
          echo "Checkpoint Path: $checkpoint_path"
          echo "Experiment Path: $experiment_dir"
          echo "Storage Path: $result_path"
          echo "============================="
          # Start the evaluation
          python3 predict_val.py $result_path --checkpoint $checkpoint_path --experiment-directory $experiment_dir --data-root $data_path  --validation-index $validation_index
          done
        done
      done
    fi
done
