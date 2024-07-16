import os
import re
from objective_configuration.reconstruction import DMODEL
from objective_helper.reconstruction import VisualizeConvergence
import argparse

"""
Sometimes... you just want to create EVERYTHING ALL AT ONCE
    But sometimes we just want to visualize on path (hence optional)
    
Since we know the exact depth, lets just model that

This shows the (validation) metrics over the iterations
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


def walk_visualization(walk_directory):
    for d, _, f in os.walk(walk_directory):
        filter_f = [x for x in f if x.startswith('metrics_val') and x.endswith('json')]
        if len(filter_f) > 0:
            # Now we have found an experiment directory
            experiment_path = d
            vis_obj = VisualizeConvergence(experiment_path, debug=debug)
            try:
                vis_obj.savefig()
            except AssertionError:
                print('Not enough files found.. Continue')


if path is not None:
    experiment_path = os.path.join(DMODEL, path)
    f = os.listdir(experiment_path)
    filter_f = [x for x in f if x.startswith('metrics_val') and x.endswith('json')]
    if len(filter_f) > 0:
        vis_obj = VisualizeConvergence(experiment_path, debug=debug)
        try:
            vis_obj.savefig()
        except AssertionError:
            print('Not enough files found.. Continue')
    else:
        walk_visualization(experiment_path)

else:
    walk_visualization(DMODEL)