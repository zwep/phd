import os
import pandas as pd
import helper.misc as hmisc
from pathlib import PurePath
import numpy as np
import argparse
from objective_helper.reconstruction import CalculateMetric
from objective_configuration.reconstruction import DRESULT, DRESULT_INFERENCE

"""
First we need to evaluate the results of course...

Then we can calculate and visualize stuff..
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '-m', type=str, help='Model path name, relative to /local_scratch/sharreve/paper/reconstruction/results')
    parser.add_argument('-perc', '-p', type=str, help='Percentage', required=False)
    parser.add_argument('-filter', '-f', type=str, required=False, help='string that must be in the dir name',
                        default='')
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--calc_us', default=False, action='store_true')

    # Parses the input
    p_args = parser.parse_args()
    pred_folder = p_args.model
    # We use this as a filter..
    sel_percentage = p_args.perc
    inference_bool = p_args.inference
    calc_us_bool = p_args.calc_us
    str_filter = p_args.filter

    if inference_bool:
        pred_folder = os.path.join(DRESULT_INFERENCE, pred_folder)
    else:
        pred_folder = os.path.join(DRESULT, pred_folder)

    if not os.path.isdir(pred_folder):
        os.makedirs(pred_folder)

    dest_json = os.path.join(pred_folder, 'metric_rebuttal.json')
    metric_table = {}
    for d, _, f in os.walk(pred_folder):
        print(pred_folder)
        filter_f = [x for x in f if x.endswith('h5')]
        # Now we have a directory containing files that can be plotted
        if len(filter_f):
            print(filter_f)
            # Check if a specific percentage is found in 'd', or if a percentage is given at all
            # TODO This is affected by sub-folder changes
            if (f'/{sel_percentage}p/' in d) or (sel_percentage is None):
                if str_filter in d:
                    print('Processing directory ', d)
                    relative_path = os.path.relpath(d, pred_folder)
                    # Split this into pieces (containing e.g. (100p, train_mixed, 2ch, 5x))
                    pp_d = PurePath(relative_path)
                    if len(pp_d.parts) == 4:
                        # Evaluation
                        percentage, dataset, anatomy, acceleration = pp_d.parts
                    elif len(pp_d.parts) == 3:
                        # Inference
                        # 25p/train_mixed/undersampled/
                        percentage, dataset, anatomy = pp_d.parts
                        acceleration = '0x'
                    else:
                        print('Warning! Not enough parts in path')
                    # Update the metric table with keys...
                    # TODO This is affected by sub-folder changes
                    _ = metric_table.setdefault(percentage, {})
                    _ = metric_table[percentage].setdefault(dataset, {})
                    _ = metric_table[percentage][dataset].setdefault(anatomy, {})
                    _ = metric_table[percentage][dataset][anatomy].setdefault(acceleration, {})

                    visualize_obj = CalculateMetric(d, ddest=pred_folder, inference_bool=inference_bool, calc_us=calc_us_bool)
                    # visualize_obj._plot()
                    metric_list = visualize_obj.calculate()
                    metric_dict = hmisc.listdict2dictlist(metric_list)
                    for i_metric, i_value in metric_dict.items():
                        print('Calculating \t ', i_metric)
                        if (i_metric != 'filename') and (i_metric != 'filename_input') and (i_metric != 'filename_target'):
                            print(i_metric, i_value[:10])
                            metric_table[percentage][dataset][anatomy][acceleration].setdefault(i_metric + '_mean', 0)
                            metric_table[percentage][dataset][anatomy][acceleration].setdefault(i_metric + '_std', 0)
                            metric_table[percentage][dataset][anatomy][acceleration][i_metric + '_mean'] = np.mean(i_value)
                            metric_table[percentage][dataset][anatomy][acceleration][i_metric + '_std'] = np.std(i_value)
                            # Lets see if this works.... it is possible.
                            metric_table[percentage][dataset][anatomy][acceleration][i_metric] = i_value

    # if os.path.isfile(dest_json):
    #     stored_metric_table = hmisc.load_json(dest_json)
    #     hmisc.update_nested(stored_metric_table, metric_table)
    #     hmisc.write_json(stored_metric_table, dest_json)
    # else:
    hmisc.write_json(metric_table, dest_json)
    print('Written json to ', dest_json)
