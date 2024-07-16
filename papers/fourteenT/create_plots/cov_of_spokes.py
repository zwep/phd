import objective_helper.fourteenT as helper_14T
import helper.plot_fun as hplotf
import helper.metric as hmetric
import matplotlib.pyplot as plt
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_DICT, \
    DDATA_KT_POWER, DPLOT_KT_POWER, PLOT_LINEWIDTH,\
    MID_SLICE_OFFSET, COIL_NAME_ORDER_TRANSLATOR, \
    DDATA_KT_VOP, DPLOT_KT_VOP
import numpy as np
import os


"""
Plot all axial slices of all kT point simulations for all coils

Then, compute the coefficient of variation for each axial slice and plot that as a function of spokes

"""


def get_masked_array(ddata, dplot, sel_coil, sel_spoke):
    """
    This retrieves the masked array and axial image for a specified coil and spoke

    :param ddata:
    :param dplot:
    :param sel_coil:
    :param sel_spoke:
    :return:
    """
    visual_obj = helper_14T.KtImage(ddata, dplot, sel_coil)
    sel_file = visual_obj._get_kt_file(sel_spoke)
    kt_array = visual_obj.get_flip_angle_map(sel_file)
    axial_img = hplotf.get_all_mid_slices(kt_array, offset=MID_SLICE_OFFSET)[-1]
    masked_array = np.abs(kt_array[visual_obj.thomas_mask_array == 1])
    return masked_array, axial_img


def get_coef_var_dict(ddata, dplot):
    """
    Here we calculate the coefficient of variation of the masked array for all spokes and coils

    :param ddata:
    :param dplot:
    :return:
    """
    coef_var_dict = {}
    for sel_spoke in range(max_spoke_kt):
        sel_spoke += 1
        axial_img_list = []
        for sel_coil in COIL_NAME_ORDER:
            _ = coef_var_dict.setdefault(sel_coil, [])
            masked_array, axial_img = get_masked_array(ddata, dplot, sel_spoke=sel_spoke, sel_coil=sel_coil)
            coef_of_var = hmetric.coefficient_of_variation(masked_array)
            coef_var_dict[sel_coil].append(coef_of_var)
            axial_img_list.append(axial_img)
    return coef_var_dict


def plot_coef_var_dict(coef_var_dict):
    """
    Plot the coefficient of variation we just calculated

    :param coef_var_dict:
    :return:
    """
    fig, ax = plt.subplots()
    for k, v in coef_var_dict.items():
        color = COLOR_DICT[k]
        coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[k]
        ax.plot(range(1, len(v) + 1), v, '-', label=coil_plot_name, color=color, linewidth=PLOT_LINEWIDTH)
#
    ax.set_xlim(0, len(v)+1)
    ax.set_xlabel('Number of ${k_T}$-points')
    ax.set_ylabel('Coefficient of Variation in flip angle')
    legend_obj = ax.legend()
    helper_14T.flush_right_legend(legend_obj)
    return fig


max_spoke_kt = 10
# Here we calculate the coef. of variation over all spokes when regularizing on Power
coef_var_dict = get_coef_var_dict(DDATA_KT_POWER, DPLOT_KT_POWER)
fig = plot_coef_var_dict(coef_var_dict)
fig.savefig(os.path.join(DPLOT_KT_POWER, 'spokes_and_coefficient_of_variation.png'))

# Here we calculate the coef. of variation over all spokes when regularizing on VOP
coef_var_dict = get_coef_var_dict(DDATA_KT_VOP, DPLOT_KT_VOP)
fig = plot_coef_var_dict(coef_var_dict)
fig.savefig(os.path.join(DPLOT_KT_VOP, 'spokes_and_coefficient_of_variation.png'))
