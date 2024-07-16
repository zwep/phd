cd /data/seb/code/pytorch_in_mri
export CUDA_VISIBLE_DEVICES="$2"
source venv/bin/activate
nnUNet_train 2d nnUNetTrainerV2 "$1" all
