import os
import reconstruction.ReadCpx as read_cpx

"""Explore how the parameter files differ...and the size of the acquired imagw"""

file_path_1 = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_03_06/V9_17911'
file_path_2 = '/media/bugger/MyBook/data/7T_scan/cardiac/2020_12_02/V9_15934'
file_list = os.listdir(file_path_2)

file_index_1 = [0, 1, 0, 1]
file_index_2 = [0, 0, 0, 0]
sel_file_index = file_index_2

sa_file = [x for x in file_list if 'sa' in x and x.endswith('cpx')][sel_file_index[0]]
fch_file = [x for x in file_list if '4ch' in x and x.endswith('cpx')][sel_file_index[1]]
tch_file = [x for x in file_list if '2ch' in x and x.endswith('cpx')][sel_file_index[2]]
cine_file = [x for x in file_list if 'cine' in x and x.endswith('cpx')][sel_file_index[3]]

for cardiac_file in [sa_file, fch_file, tch_file, cine_file]:
    print('\n\n #### CINE FILE #### \n\n')
    print('file name ', cardiac_file)
    cpx_obj = read_cpx.ReadCpx(os.path.join(file_path_2, cardiac_file))
    par_file = cpx_obj.get_par_file()
    cpx_img = cpx_obj.get_cpx_img()
    import helper.misc as hmisc
    hmisc.print_dict(par_file)
    print(cpx_img.shape)