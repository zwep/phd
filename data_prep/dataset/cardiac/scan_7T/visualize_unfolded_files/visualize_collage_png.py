import scipy.io
import helper.misc as hmisc
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import helper.plot_class as hplotc

from data_prep.dataset.cardiac.scan_7T.visualize_unfolded_files import FileGatherer
import collections


i_v_number = 'V9_16655'
# Creating collages.. from scratch.. per V-number
data_dir = '/media/bugger/MyBook/data/7T_data/cardiac_cine_mat_png'
for i_v_number in os.listdir(data_dir):
    d_vnumber = os.path.join(data_dir, i_v_number)
    file_list_png = []
    for d, _, f in os.walk(d_vnumber):
        print(d)
        temp_file_list_png = [os.path.join(d, x) for x in f if x.endswith('png') and (not x.startswith('collage'))]
        file_list_png.extend(temp_file_list_png)

    n_files = len(file_list_png)
    multp_six = n_files // 6 + int((n_files % 6) > 0)
    n_max_files = multp_six * 6
    n_difference = n_max_files - n_files
    for _ in range(n_difference):
        file_list_png.append([])

    n_files = len(file_list_png)
    multp_six = n_files // 6
    counter = -1
    for index_range in np.split(np.arange(n_files), multp_six):
        plot_array = []
        file_string = ''
        for ii in index_range:
            sel_file = file_list_png[ii]
            if len(sel_file):
                base_name = re.findall('(sa|tra|4ch|2ch)', sel_file)[0]
                # file_png = os.path.join(model_dir_path, sel_file)
                png_array = hmisc.load_array(sel_file)
                # print(png_array.shape)
                plot_array.append(png_array)
                file_string += base_name + "_"
        dest_file_name = os.path.join(d_vnumber, 'collage_' + file_string[:-1] + '.png')
#        plot_array = np.stack(plot_array)
        fig_obj = hplotc.ListPlot([plot_array], debug=True, cmap='rgb', sub_col_row=(3, 2), ax_off=True, wspace=0, hspace=0,
                                  figsize=(10, 15))
        fig_obj.figure.savefig(dest_file_name, bbox_inches='tight', pad_inches=0.0)
