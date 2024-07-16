from objective_configuration.segment7T3T import CLASS_INTERPRETATION, COLOR_DICT, \
    COLOR_DICT_RGB, CMAP_ALL, MY_CMAP, get_path_dict
import os
import re
"""
Lets check how big the problem is

"""

label_files_acdc = set(os.listdir(get_path_dict('acdc')['dlabel']))
label_files_mm1a = set(os.listdir(get_path_dict('mm1a')['dlabel']))
label_files_mm1b = set(os.listdir(get_path_dict('mm1b')['dlabel']))

# Check for any nnunet thing.. if
ddata_nnraw = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data'
for i_problem in os.listdir(ddata_nnraw):
    problem_dir = os.path.join(ddata_nnraw, i_problem)
    print(problem_dir)
    if os.path.isdir(problem_dir):
        # Taking label dir, because that one does not have that annoying _0000 appendix
        label_dir = os.path.join(problem_dir, 'labelsTr')
        label_file_list = os.listdir(label_dir)
        label_file_list = [re.sub('(^.*to_)', '', x) for x in label_file_list]
        label_files_problem = set(label_file_list)
        n_acdc = len(label_files_problem.intersection(label_files_acdc))
        n_mm1a = len(label_files_problem.intersection(label_files_mm1a))
        n_mm1b = len(label_files_problem.intersection(label_files_mm1b))
        print(f'Overlap acdc {n_acdc} mm1a {n_mm1a} mm1b {n_mm1b}')
