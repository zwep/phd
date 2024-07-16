import scipy.io
import shutil
import re
import os
import numpy as np
import helper.reconstruction as hrecon
import reconstruction as recon

"""
We have obtained unsorted cardiac data
"""


class ParseReconList:
    def __init__(self, file_path):
        base_name, ext = os.path.splitext(file_path)
        self.file_path = base_name + '.list'

    def read_text_file(self):
        with open(self.file_path, 'r') as f:
            read_lines = f.readlines()

        return read_lines


if __name__ == "__main__":
    # Here we do a name change...
    # ddata = '/media/bugger/MyBook/data/7T_data/cardiac_unsorted_data'
    ddata = '/media/bugger/B7DF-4571/2022_02_28/ca_32384/matdata'
    for d, _, file_list in os.walk(ddata):
        # print(d)
        for x in file_list:
            # This should do the trick I suppose...
            x_orig = x
            x = re.sub('_data', '', x)
            x = re.sub('_label', '', x)
            source_file = os.path.join(d, x_orig)
            target_file = os.path.join(d, x)
            print(f"Moving {source_file} -> {target_file}")
            shutil.move(source_file, target_file)


    # That should do the trick for now...
#    ddata = '/media/bugger/MyBook/data/7T_data/cardiac_unsorted_data/V9_13518'
    ddata = '/media/bugger/B7DF-4571/2022_02_28/ca_32383/matdata'
    file_list = os.listdir(ddata)
    sel_file = os.path.join(ddata, file_list[2])
    print("Loading ", sel_file)

    mat_obj = scipy.io.loadmat(sel_file)
    unsorted_spokes = mat_obj['unsorted_data'][0][0]
    unsorted_spokes.shape

    parse_obj = ParseReconList(sel_file)
    parse_obj.read_text_file()

#
# zz = scipy.io.loadmat(ddata_unsorted)['reconstructed_data']
# hplotc.SlidingPlot(zz[:, ::24])
#
#
# # The .sin file was not exported.... in the .par file
# n_coil = 24
# n_dyn = 80
# n_spokes = 66
# center_index = mat_obj.shape[0]//2
#
# coil_signal = []
# for i_coil in range(n_coil):
#     coil_info = mat_obj[center_index, i_coil::n_coil]
#     coil_signal.append(coil_info)
#
# coil_signal = np.array(coil_signal)
# n_acquired_spokes = coil_signal.shape[1]
#
# TR = 4.5 * 10 ** -3
# TR = 29.28 / n_acquired_spokes # 29.28 is the total scan duration
#
# time_range = np.arange(n_acquired_spokes) * TR
# fig, ax = plt.subplots(8)
# for i_spoke in range(0, n_spokes, 10):
#     for ii, sel_coil in enumerate(coil_signal[-8:]):
#         # ax[ii].plot(time_range, np.abs(sel_coil))
#         ax[ii].plot(time_range[i_spoke::n_spokes], np.abs(sel_coil)[i_spoke::n_spokes], 'r-o', alpha=0.2)
