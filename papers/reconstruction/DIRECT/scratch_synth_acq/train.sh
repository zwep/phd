#!/bin/bash


# Get the directory of the current script
code_dir=${HOME}/local_scratch/code/pytorch_in_mri/papers/reconstruction/DIRECT
# Import functions from bash_functions.sh using a relative path
source "$code_dir/bash_functions.sh"

username=$(whoami)

if [[ $username == "sharreve" ]]; then
    HOME=${HOME}"/local_scratch"
    DDATA="${HOME}/mri_data/cardiac_full_radial"
    PRETRAINED="${HOME}/model_run/direct"
    MODEL_RUN="${HOME}/model_run/direct"
else
    echo "Invalid username."
    exit 1
fi

echo "DDATA: $DDATA"
echo "MODEL_RUN: $MODEL_RUN"

# Setting default values...
percentage=-1
percentage_model=-1
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
    --percentage_model|-p_model)
      percentage_model="$2"
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
    base_dir="${PRETRAINED}/varnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  base_conjgradnet)
    # base_ckpt="${PRETRAINED}/base_conjgradnet_SCRATCH_SYNTH/100p/train_mixed/model_4999.pt"
    base_dir="${PRETRAINED}/base_conjgradnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  rim)
    # base_ckpt="${PRETRAINED}/rim/model_89000.pt"
    base_dir="${PRETRAINED}/rim_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  multidomainnet)
    # base_ckpt="${PRETRAINED}/multidomainnet/model_50000.pt"
    base_dir="${PRETRAINED}/multidomainnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  xpdnet)
    # base_ckpt="${PRETRAINED}/xpdnet_SCRATCH_SYNTH/100p/train_mixed/model_4999.pt"
    base_dir="${PRETRAINED}/xpdnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  recurrentvarnet)
    # base_ckpt="${PRETRAINED}/recurrentvarnet/model_148500.pt"
    base_dir="${PRETRAINED}/recurrentvarnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  jointicnet)
    # base_ckpt="${PRETRAINED}/jointicnet/model_43000.pt"
    base_dir="${PRETRAINED}/jointicnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  kikinet)
    # base_ckpt="${PRETRAINED}/kikinet/model_44500.pt"
    base_dir="${PRETRAINED}/kikinet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  base_iterdualnet)
    # base_ckpt="${PRETRAINED}/base_iterdualnet/model_33500.pt"
    base_dir="${PRETRAINED}/base_iterdualnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  unet | unet_test)
    # base_ckpt="${PRETRAINED}/unet_SCRATCH_SYNTH/100p/train_mixed/model_4999.pt"
    base_dir="${PRETRAINED}/unet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
    ;;
  lpdnet)
    # base_ckpt="${PRETRAINED}/lpdnet/model_96000.pt"
    base_dir="${PRETRAINED}/lpdnet_SCRATCH_SYNTH/${percentage_model}p/train_mixed"
    base_ckpt=$(find_model_file "${base_dir}")
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
  # output_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH_ACQ/${percentage}p"
  output_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH_ACQ/${percentage_model}p"
  # train_oot="${DDATA}/${anatomy}/train_${percentage}/input"
  if ${DEBUG}; then
    # config_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH_ACQ/config/config_${percentage}p_${anatomy}_debug.yaml"
    config_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH_ACQ/config/config_${percentage}p_${anatomy}_debug.yaml"
    val_root="${DDATA}/${anatomy}/validation_1/input"
    train_root="${DDATA}/${anatomy}/validation_1/input"
  else
    # config_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH_ACQ/config/config_${percentage}p_${anatomy}.yaml"
    config_path="${MODEL_RUN}/${model_name}_SCRATCH_SYNTH_ACQ/config/config_${percentage}p_${anatomy}.yaml"
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
           --num-workers 10
done

