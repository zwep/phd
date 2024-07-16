#!/bin/bash

latest_job_path="/home/bme001/20184098/latest_job_id.log"
latest_job_id=$(tail "$latest_job_path")

show_all=false
# Replace it with a possible input arguent..
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -j|--job_id) latest_job_id="$2"; shift ;;
        -a|--all) show_all=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

latest_job_file="/home/bme001/20184098/data/report/output_$latest_job_id.out"

echo "Showing result from this job ID: $latest_job_id"
if [ "$show_all" = true ] ; then
    cat "$latest_job_file"
else
    tail "$latest_job_file"
fi

