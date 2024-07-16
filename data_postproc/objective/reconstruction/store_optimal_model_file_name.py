import numpy as np
from objective_helper.reconstruction import VisualizeConvergence
import os
from objective_configuration.reconstruction import DMODEL
import matplotlib.pyplot as plt
import argparse

"""
I want to check if Im not using a model file that has been trained TOO MUCH
"""

parser = argparse.ArgumentParser()
parser.add_argument('-path', default=None,
                    type=str,
                    nargs='?',  # This makes it optional
                    help='Provide the name of the directory that we want to post process')
parser.add_argument('--debug', default=False, action='store_true')

p_args = parser.parse_args()
path = p_args.path
debug = p_args.debug


def walk_optimal_model_pt(walk_directory):
    for d, _, f in os.walk(walk_directory):
        filter_f = [x for x in f if x.startswith('metrics_val') and x.endswith('json')]
        if len(filter_f) > 0:
            # Now we have found an experiment directory
            experiment_path = d
            print(experiment_path)
            vis_obj = VisualizeConvergence(experiment_path)
            try:
                optimal_nr = vis_obj.write_optimal_model_pt_file()
                print(optimal_nr)
            except AssertionError:
                print('Not enough files found.. Continue')


if path is not None:
    # If a specified path is given, it is always relative to DMODEL
    # It needs to be a folder containing metrics_val
    # If it doesnt, we will simply walk that directory
    experiment_path = os.path.join(DMODEL, path)
    f = os.listdir(experiment_path)
    filter_f = [x for x in f if x.startswith('metrics_val') and x.endswith('json')]
    if len(filter_f) > 0:
        vis_obj = VisualizeConvergence(experiment_path, debug=debug)
        try:
            optimal_nr = vis_obj.write_optimal_model_pt_file()
            print(optimal_nr)
        except AssertionError:
            print('Not enough files found.. Continue')
    else:
        walk_optimal_model_pt(experiment_path)
else:
    # If nothing is given, we will run evrything
    walk_optimal_model_pt(DMODEL)
