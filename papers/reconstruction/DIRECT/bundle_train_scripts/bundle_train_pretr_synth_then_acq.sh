#!/usr/bin/bash

model_name=unet
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

# Activate the direct environment
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate /local_scratch/sharreve/anaconda3/envs/venv

cd ${HOME}/local_scratch/code/pytorch_in_mri
# cd ${HOME}/code/pytorch_in_mri

: '
Below are the Calgary pretrained models

'

# First train the PRETR_SYNTH model

python data_prep/objective/reconstruction/pretr_synth/create_config_files.py -path $model_name
./papers/reconstruction/DIRECT/pretr_synth/train.sh -m $model_name -p 100

# Then train the PRETR_SYNTH_ACQ model
python data_prep/objective/reconstruction/pretr_synth_acq/create_config_files.py -path $model_name
./papers/reconstruction/DIRECT/pretr_synth_acq/train.sh -m $model_name -p 100 -p_model 100
#