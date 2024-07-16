import scipy.io
import helper.misc as hmisc
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import helper.plot_class as hplotc

from data_prep.dataset.cardiac.scan_7T.visualize_unfolded_files import FileGatherer
import collections


data_dir = '/media/bugger/MyBook/data/7T_data/cardiac_cine_mat'
plot_dir = '/media/bugger/MyBook/data/7T_data/cardiac_cine_mat_png'
file_gather_obj = FileGatherer.FileGather(data_dir)
plot_file_obj = FileGatherer.PlotFileList(file_list=file_gather_obj.file_list_ext,
                                          source_dir=data_dir, dest_dir=plot_dir)

plot_file_obj.plot_all_files()


