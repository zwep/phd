"""
We need the sin files.. chekc the voxel sizes for Sina Yasmina
"""

import os

ddata = '/media/bugger/MyBook/data/7T_scan/cardiac'
ddata_sa = '/media/bugger/MyBook/data/7T_data/unfolded_sa_filtered'
filtered_sa_files = [os.path.splitext(x)[0] for x in os.listdir(ddata_sa) if x.endswith('nrrd')]

sa_sin_files = []
for d, _, f in os.walk(ddata):
    filter_f = [x for x in f if 'sin' in x and 'sa' in x and not 'radial' in x]
    if filter_f:
        # print(filter_f)
        for i_sin in filter_f:
            sin_file = os.path.join(d,i_sin)
            with open(sin_file, 'r') as file_obj:
                sin_list = file_obj.readlines()

            file_name = os.path.splitext(i_sin)[0]
            if file_name in filtered_sa_files:
                voxel_size_line = [x for x in sin_list if 'voxel_sizes' in x][0]
                voxel_size_pretty = ', '.join(voxel_size_line.strip().split(':')[-1].strip().split())
                print(file_name, '\t', voxel_size_pretty)