#!/bin/bash

# This command is a wrapper around sbatch
# Using this we can easily extract the job id from an sbatch run

while getopts ":j:" opt; do
  case "$opt" in
  # This gets the job id
    j) OPTIONAL_VAR="$OPTARG"
       break;;
    ?) ;;
  esac
done



if [ -z "$OPTIONAL_VAR" ]; then
    sbr="$(sbatch ~/code/pytorch_in_mri/slurm/basic_execution.sh "$@")"
else
    # shift out the processed options
    shift $((OPTIND-1))
    sbr="$(sbatch --dependency=afterok:$OPTIONAL_VAR ~/code/pytorch_in_mri/slurm/basic_execution.sh "$@")"
fi    

if [[ "$sbr" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
else
    echo "sbatch failed"
fi
