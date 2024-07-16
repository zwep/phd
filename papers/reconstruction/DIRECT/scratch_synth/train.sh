#!/bin/bash

# Help message
display_help() {
  echo "Usage: script_name.sh [options]"
  echo "    Not doing anything"
}
# I need to parse some input arguments...
# These are
#   - Model name (unet_parallel, varnet_parallel, ...)
#   - Training Percentage (25, 50, 75, 100)

username=$(whoami)

if [[ $username == "sharreve" ]]; then
    HOME=${HOME}"/local_scratch"
    DDATA="${HOME}/mri_data/cardiac_synth_7T/direct_synth"
    MODEL_RUN="${HOME}/model_run/direct"
elif [[ $username == "20184098" ]]; then
    DDATA="/home/bme001/20184098/data/direct_synth"  # This one does not exist
    MODEL_RUN="/home/bme001/20184098/model_run/direct"
    module load cuda11.5/toolkit
    echo "Cuda visible devices " $CUDA_VISIBLE_DEVICES
else
    echo "Invalid username."
    exit 1
fi

echo "DDATA: $DDATA"


# Setting default values...
percentage=-1
model_name='unet'
DEBUG=false
num_gpu=1
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --model_name|-m)
      model_name="$2"
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
    --num_gpu|-n)
      num_gpu="$2"
      shift # Shift past the argument name
      shift # Shift past the argument value
      ;;
    --debug)
      DEBUG=true
      echo "Debug is set to true"
      shift
      ;;
    *) # Handle unrecognized arguments
      echo "Unrecognized argument: $key"
      shift # Shift past the argument
      ;;
  esac
done


# Now I can execute direct train...
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate direct

echo "Activated Direct"

name="train"

# Loop over given list..
for anatomy in 'mixed'; do
  echo ""
  echo "   ==============================================      "
  echo ""
  echo "anatomy ${anatomy}"
  # Update the name with the anatomy
  name="${name}_${anatomy}"
  output_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH/${percentage}p"
  if ${DEBUG}; then
    config_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH/config/config_${percentage}p_${anatomy}_debug.yaml"
    val_root="${DDATA}/${anatomy}/validation_1/input"
    train_root="${DDATA}/${anatomy}/validation_1/input"
  else
    config_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH/config/config_${percentage}p_${anatomy}.yaml"
    val_root="${DDATA}/${anatomy}/validation/input"
    train_root="${DDATA}/${anatomy}/train/input"
  fi

  # Create result path if it doesnt exist
  if [ ! -d "$output_path" ]; then
    mkdir -p "$output_path"
#  else
  #  rm -r ${output_path:?}/*
  fi

  echo 'Name            ' $name
  echo 'Config path     ' $config_path
  echo 'Output path     ' $output_path
  echo 'Training root   ' $train_root
  echo 'Validation root ' $val_root

  direct train $output_path \
           --training-root $train_root \
           --validation-root $val_root  \
           --name $name \
           --cfg $config_path \
           --num-gpus $num_gpu \
           --num-workers 10
done

