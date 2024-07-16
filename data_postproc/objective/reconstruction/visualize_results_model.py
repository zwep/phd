import os
import getpass

if getpass.getuser() == 'sharreve':
    import matplotlib
    matplotlib.use('Agg')

from pathlib import PurePath
import argparse
from objective_helper.reconstruction import VisualizeFolder
from objective_configuration.reconstruction import DRESULT, DRESULT_INFERENCE, DRESULT_RETRO, DRETRO


"""

Given the name of a model/folder (e.g. kikinet_resume) it produdes images of all the .h5 files that are nseted in there
and stored them under appropriate names

Structure is like this

model
    percentage_1
        dataset_1
            2ch
                5x
                10x
            4ch
                ..
            sa
                ..
            transverse
                ..
        ...
        dataset_m
            ..
    percentage_...
        
        
    

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '-m', type=str, help='Model path name, relative to /local_scratch/sharreve/paper/reconstruction/results')
    parser.add_argument('-perc', '-p', type=str, help='Percentage', required=False)
    parser.add_argument('-filter', '-f', type=str, required=False, help='string that must be in the dir name', default='')
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--retro', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')

    # Parses the input
    p_args = parser.parse_args()
    debug = p_args.debug
    retro_bool = p_args.retro
    inference_bool = p_args.inference
    percentage = p_args.perc
    pred_folder = p_args.model
    str_filter = p_args.filter
    # percentage = p_args.perc

    if inference_bool:
        pred_folder = os.path.join(DRESULT_INFERENCE, pred_folder)
    elif retro_bool:
        pred_folder = os.path.join(DRESULT_RETRO, pred_folder)
    else:
        pred_folder = os.path.join(DRESULT, pred_folder)

    if debug:
        print('Selected folder', pred_folder)

    for d, _, f in os.walk(pred_folder):
        filter_f = [x for x in f if x.endswith('h5')]
        # Now we have a directory containing files that can be plotted
        if debug:
            print(d, len(filter_f))
        if len(filter_f):
            if debug:
                print(f'Number of files found {len(filter_f)}')
            # TODO This is affected by sub-folder changes
            if (f'/{percentage}p/' in d) or (percentage is None):
                if str_filter in d:
                    # Get the path relative to the results path
                    relative_path = os.path.relpath(d, pred_folder)
                    # Split this into pieces
                    pp_d = PurePath(relative_path)
                    str_appendix = '_eval:'.join(pp_d.parts)
                    # I could do this differently..
                    # For example create a new directory and store it there. But fuckit for now..
                    visualize_obj = VisualizeFolder(d, ddest=pred_folder, file_str_appendix=str_appendix, retro_bool=retro_bool,
                                                    inference_bool=inference_bool, debug=debug, calc_us=False)
                    print('Visualizing path ', d)
                    print('Storage path ', pred_folder)
                    print('Str appendix path ', str_appendix)
                    print()
                    print()
                    visualize_obj.plot()
#
# import os
# import h5py
# import numpy as np
#
# # Define the directory path
# directory_path = '/home/sharreve/local_scratch/mri_data/cardiac_radial_inference'
#
# # Function to process a single HDF5 file
# def process_h5_file(file_path):
#     with h5py.File(file_path, 'r+') as h5_file:
#         if 'kspace' in h5_file:
#             reversed_kspace = np.array(h5_file['kspace'])[:, ::-1, ::-1, :]
#             # print(kspace_data.shape)
#             # reversed_kspace = np.transpose(kspace_data, axes=(2, 1, *range(2, kspace_data.ndim)))
#             del h5_file['kspace']
#             h5_file.create_dataset('kspace', data=reversed_kspace)
#
# # Walk through the directory and process .h5 files
# for root, dirs, files in os.walk(directory_path):
#     for file_name in files:
#         if file_name.endswith('.h5'):
#             file_path = os.path.join(root, file_name)
#             print(f"Processing {file_path}")
#             process_h5_file(file_path)
#             print(f"Processed {file_path}")
#
# print("Processing complete.")