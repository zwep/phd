import os
import helper.misc as hmisc
import helper.array_transf as harray
import numpy as np
import helper.plot_class as hplotc
import helper.plot_fun as hplotf

"""
derp.
"""

fontsize = 14

ddata = '/home/bugger/Documents/paper/inhomogeneity removal/rebuttal/latest_version/plot_arrays/model_results/patient_7T'
file_list = os.listdir(ddata)
file_list_filter = [x for x in file_list if x.endswith('npy')]


for sel_file in file_list_filter:
    # Define paths
    sel_file_no_ext = hmisc.get_base_name(sel_file)
    sel_file_path = os.path.join(ddata, sel_file)
    sel_text_path = os.path.join(ddata, sel_file_no_ext + "_text.txt")
    if 'crop' not in sel_file_no_ext:
        sel_metric_path = os.path.join(ddata, sel_file_no_ext + "_metric.txt")
        with open(sel_metric_path, 'r') as f:
            temp = f.read()
            homogeneity_energy_list = [x.strip() for x in temp.split(',')]
            homogeneity_energy_list = [eval(x + ', ' + y) for x, y in
                                       zip(homogeneity_energy_list[::2], homogeneity_energy_list[1::2])]

    dest_file_path = os.path.join(ddata, sel_file_no_ext + '.png')


    # Load stuff
    plot_array = np.load(sel_file_path)

    with open(sel_text_path, 'r') as f:
        temp = f.read()
        text_box_names = [x for x in temp.split(',')]



    patch_size = plot_array.shape[-1] // 10
    plot_array = harray.scale_minmax(plot_array, axis=(-2, -1))
    vmax_list = [(0, harray.get_proper_scaled_v2(x, (patch_size, patch_size), patch_size // 2)) for x in plot_array]


    """         Plot the plot array       """
    fig_obj = hplotc.ListPlot(plot_array[None], ax_off=True, hspace=0, wspace=0, vmin=[vmax_list], figsize=(30, 10))
    # fig_obj = hplotc.ListPlot(plot_array[None], ax_off=True, hspace=0, wspace=0,  figsize=(30, 10))
    fig = fig_obj.figure

    for i, i_text_box_name in enumerate(text_box_names):
        hplotf.add_text_box(fig, i, i_text_box_name, height_offset=0, position='top')

    # Now add the bottom text boxes...
    for ii, (i_hom, i_energ) in enumerate(homogeneity_energy_list):
        i_hom = "%0.2f" % i_hom
        i_energ = "%0.2f" % i_energ
        hplotf.add_text_box(fig, ii, f'H:{i_hom}        E:{i_energ}', height_offset=0, position='bottom', fontsize=fontsize)


    fig.savefig(dest_file_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    hplotc.close_all()
