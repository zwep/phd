import os

"""
Returns an overview of the amount of images....
"""

ddir = '/local_scratch/sharreve/mri_data/vanLier_Prostaat_T2'

shape_list_dict = {}
for d, _, f in os.walk(ddir):
    MRL_MRI = os.path.basename(d)
    patient_ID = os.path.basename(os.path.dirname(d))

    filter_list = [x for x in f if x.endswith('.dcm')]
    n_files = len(filter_list)
    if n_files:
        temp_key = MRL_MRI + patient_ID
        shape_list_dict.setdefault(temp_key, n_files)

list_keys = sorted(list(shape_list_dict.keys()))
n_space = 50
print('Key', ' ' * 47, ' Number of files')

for i_key in list_keys:
    print(i_key, ' ' * (n_space - len(i_key)), shape_list_dict[i_key])

