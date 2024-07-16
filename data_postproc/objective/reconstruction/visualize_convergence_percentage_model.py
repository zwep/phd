import os
import re
from objective_configuration.reconstruction import DMODEL, TYPE_NAMES, MODEL_NAMES
from objective_helper.reconstruction import VisualizePercentageConvergence
import argparse
import matplotlib.pyplot as plt

"""
This visualizes the last validation metric over the varying percentages
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
        experiment_path = os.path.join(DMODEL, d)
        if os.path.isdir(experiment_path):
            vis_obj = VisualizePercentageConvergence(experiment_path, debug=debug)
            fig_obj = vis_obj.visualize_metrics()
            fig_obj.savefig(os.path.join(experiment_path, 'validation_conv.png'))
            plt.close()


if path is not None:
    experiment_path = os.path.join(DMODEL, path)
    vis_obj = VisualizePercentageConvergence(experiment_path, debug=debug)
    fig_obj = vis_obj.visualize_metrics()
    fig_obj.savefig(os.path.join(path, 'validation_conv.png'))
else:
    walk_visualization(DMODEL)
#
# # Gawd fucking dammit...
# # Indexing is super annoying
# path = os.path.join(DMODEL, path)
# visual_obj = VisualizePercentageConvergence(path, debug=debug)
# fig_obj = visual_obj.visualize_metrics()
# fig_obj.savefig(os.path.join(path, 'validation_conv.png'))