import pandas as pd
import helper.reconstruction as hrecon
import os

"""
Find resolution with stuff..
"""
ddata = '/media/bugger/MyBook/data/7T_scan/cardiac'
overview_csv = '/media/bugger/MyBook/data/misc/overview_new_batch.csv'
overview_dataframe = pd.read_csv(overview_csv)
unique_mat_files = list(set(overview_dataframe['mat files']))
mat_name_resolution = []
for mat_name in unique_mat_files:
    file_name = mat_name + '.sin'
    found_file = False
    for d, _, f in os.walk(ddata):
        if found_file:
            break
        else:
            if file_name in f:
                print(d, file_name)
                sel_file = os.path.join(d, file_name)
                voxel_sizes = hrecon.get_key_from_sin_file(sel_file, 'voxel_sizes')
                found_file = True
    if found_file is False:
        voxel_sizes = 'None'
    mat_name_resolution.append([mat_name, voxel_sizes])


with open(os.path.join('/media/bugger/MyBook/data/misc/name_resolution_match.txt'), 'w') as f:
    write_string = '\n'.join(['\t'.join(x) for x in mat_name_resolution])
    f.write(write_string)


"""
Two of these werent able to be found..

Manually I checked the .par file of the .cpx file and got

v9_05032022_1259040_11_2_saV4	FOV (ap,fh,rl) [mm]	60.00 238.69 300.95
v9_24012021_1259030_14_2_sa_fullfovV4	FOV (ap,fh,rl) [mm]	88.00 330.61 263.46

So it is something..
"""