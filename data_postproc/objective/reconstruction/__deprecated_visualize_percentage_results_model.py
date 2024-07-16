import getpass

if getpass.getuser() == 'sharreve':
    import matplotlib
    matplotlib.use('Agg')

import os
from pathlib import PurePath
import argparse
from objective_helper.reconstruction import VisualizeFolder
from objective_configuration.reconstruction import DRESULT, PERCENTAGE_LIST
import re

"""
Here we visualize one input then X percentage predictions (0, 25, 50, 75, 100) and the target



?? What is the difference between this one and visualize_results_per_percentage
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '-p', type=str,
                        help='Model path name, relative to /local_scratch/sharreve/paper/reconstruction/results')

    # Parses the input
    p_args = parser.parse_args()
    pred_folder = p_args.path
    # Only walk over the 0% folder...
    pred_folder = os.path.join(DRESULT, pred_folder, '0p')
    for d, _, f in os.walk(pred_folder):
        filter_f = [x for x in f if x.endswith('h5')]
        # Now we have a directory containing files that can be plotted
        if len(filter_f):
            # Get the path relative to the results path
            relative_path = os.path.relpath(d, pred_folder)
            # Split this into pieces
            pp_d = PurePath(relative_path)
            str_appendix = '_eval:'.join(pp_d.parts)
            # I could do this differently..
            # For example create a new directory and store it there. But fuckit for now..
            print(d)
            anatomy = os.path.basename(os.path.dirname(d))
            acc = os.path.basename(d)
            visualize_obj = VisualizeFolder(d, ddest=pred_folder, file_str_appendix=str_appendix)
            n_images = visualize_obj.us_array.shape[0]
            for ii in range(n_images):
                plot_array = [visualize_obj.us_array[ii]]
                d_sub = re.sub('pretrained', 'train_mixed', d)
                # This here is pretty stupid... loads in a lot of data that will never be used
                for i_perc in PERCENTAGE_LIST:
                    d_sub_perc = re.sub('0p', str(i_perc) + 'p', d_sub)
                    visualize_obj = VisualizeFolder(d_sub_perc, ddest=pred_folder, file_str_appendix=str_appendix)
                    plot_array.append(visualize_obj.pred_array[ii])

                plot_array.append(visualize_obj.target_array[ii])

                import helper.plot_class as hplotc
                subtitle_list = ['input'] + [str(x) + '%' for x in PERCENTAGE_LIST] + ['target']
                fig_obj = hplotc.ListPlot([x[::-1] for x in plot_array], col_row=(len(plot_array), 1), subtitle=[[x] for x in subtitle_list],
                                          figsize=(30, 10), ax_off=True)
                fig_obj.figure.savefig(os.path.join(os.path.dirname(pred_folder), f'last_image_pred_{anatomy}_{acc}_{ii}.png'), bbox_inches='tight', pad_inches=0)
