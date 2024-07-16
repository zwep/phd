"""
Tedious command creation
"""

import os
dd = '/local_scratch/sharreve/model_run/direct'
for model_name in os.listdir(dd):
    if 'parallel' in model_name:
        create_conf = f"python data_prep/objective/reconstruction/parallel/create_config_files.py -p {model_name}"
        run_train = f"./papers/reconstruction/DIRECT/parallel/train.sh -m {model_name} -p 25"
        print(create_conf)
        print(run_train)