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
    PRETRAINED="${HOME}/mri_data/pretrained_networks/direct"
    MODEL_RUN="${HOME}/model_run/direct"
elif [[ $username == "20184098" ]]; then
    DDATA="/home/bme001/20184098/data/direct_synth"  # This one does not exist
    MODEL_RUN="/home/bme001/20184098/model_run/direct"
    PRETRAINED="/home/bme001/20184098/data/pretrained_networks/direct"
    module load cuda11.5/toolkit
    echo "Cuda visible devices " $CUDA_VISIBLE_DEVICES
else
    echo "Invalid username."
    exit 1
fi

echo "DDATA: $DDATA"
echo "MODEL_RUN: $MODEL_RUN"
echo "PRETRAINED: $PRETRAINED"


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
    --num_gpu|-n)
      num_gpu="$2"
      shift # Shift past the argument name
      shift # Shift past the argument value
      ;;
    --help|-h) # Handle --help and -h arguments
      display_help
      exit 0
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

# I know.. this is a bit stupud
# But it might work
case "$model_name" in
  varnet)
    base_ckpt="${PRETRAINED}/varnet/model_4000.pt"
    ;;
  base_conjgradnet)
    base_ckpt="${PRETRAINED}/base_conjgradnet/model_55500.pt"
    ;;
  rim)
    base_ckpt="${PRETRAINED}/rim/model_89000.pt"
    ;;
  multidomainnet)
    base_ckpt="${PRETRAINED}/multidomainnet/model_50000.pt"
    ;;
  xpdnet)
    base_ckpt="${PRETRAINED}/xpdnet/model_16000.pt"
    ;;
  recurrentvarnet)
    base_ckpt="${PRETRAINED}/recurrentvarnet/model_148500.pt"
    ;;
  jointicnet)
    base_ckpt="${PRETRAINED}/jointicnet/model_43000.pt"
    ;;
  kikinet)
    base_ckpt="${PRETRAINED}/kikinet/model_44500.pt"
    ;;
  base_iterdualnet)
    base_ckpt="${PRETRAINED}/base_iterdualnet/model_33500.pt"
    ;;
  unet | unet_test)
    base_ckpt="${PRETRAINED}/unet/model_10000.pt"
    ;;
  lpdnet)
    base_ckpt="${PRETRAINED}/lpdnet/model_96000.pt"
    ;;
  *)
    echo "Invalid model_name. Please provide a valid model_name."
    exit 1
    ;;
esac

# Now I can execute direct train...
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate direct


echo "Selected model path: $base_ckpt"

name="train"

# Loop over given list..
for anatomy in 'mixed'; do
  echo ""
  echo "   ==============================================      "
  echo ""
  echo "anatomy ${anatomy}"
   # Update the name with the anatomy
  name="${name}_${anatomy}"
  output_path="${MODEL_RUN}/${model_name}_PRETR_SYNTH/${percentage}p"
  # train_oot="${DDATA}/${anatomy}/train_${percentage}/input"
  if ${DEBUG}; then
    config_path="${MODEL_RUN}/${model_name}_PRETR_SYNTH/config/config_${percentage}p_${anatomy}_debug.yaml"
    val_root="${DDATA}/${anatomy}/validation_1/input"
    train_root="${DDATA}/${anatomy}/validation_1/input"
  else
    config_path="${MODEL_RUN}/${model_name}_PRETR_SYNTH/config/config_${percentage}p_${anatomy}.yaml"
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
  echo 'Checkpoint path ' $base_ckpt
  echo 'Training root   ' $train_root
  echo 'Validation root ' $val_root

  direct train $output_path \
           --initialization-checkpoint $base_ckpt \
           --training-root $train_root \
           --validation-root $val_root  \
           --name $name \
           --cfg $config_path \
           --num-gpus $num_gpu \
           --num-workers 10 \
           --dont-load-sense-model # We need this in order to prevent the initial loading of the sense model...

done

