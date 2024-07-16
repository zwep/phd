"""
Because this studid thing takes too long we need to precalculate all these flipping shim settings

"""
import os
os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "12"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "12"  # export NUMEXPR_NUM_THREADS=6


import data_generator.InhomogRemoval as data_gen
import sys
import pandas as pd
import numpy as np

dir_data = '/local_scratch/sharreve/mri_data/registrated_h5'

data_type = 'train'
for _ in range(10):
    for data_type in ['train', 'test', 'validation']:
        shim_dir = os.path.join(dir_data, data_type, 'shimsettings')
        if not os.path.isdir(shim_dir):
            os.makedirs(shim_dir)

        print(f"========== {data_type} ==========\n\n\n")
        data_gen_obj = data_gen.PreComputerShimSettings(ddata=dir_data, dataset_type=data_type,
                                                        objective_shim='signal_se', file_ext='h5',
                                                        shuffle=False)
        counter = 0
        for item in data_gen_obj:
            counter += 1
            print("COUNTER ", counter)
            pass
