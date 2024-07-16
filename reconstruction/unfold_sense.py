# Tika nad PyPDF2 dont work that easily
# Used an online service..

"""
I want to analyse the Abstract results... but the format is just horrible..
Either use some different tool.. or just forget about it.
"""

import skimage.transform as sktransform
import scipy.signal
import os
import skimage.transform as sktrans
import numpy as np
import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc

import helper.plot_fun as hplotf
import importlib


# Unfolding...
def unfold(folded_image, reference_img, folding_factor):
    # Make complex here optional based on the input..
    unfolded_image = np.zeros(reference_img.shape[-2:], dtype=complex)
    n_c, n_x, n_y = reference_img.shape
    n_x_fold = int(n_x / folding_factor)
    for i_x in range(n_x_fold):
        for i_y in range(n_y):
            temp_signal = folded_image[:, i_x, i_y]
            temp_sens = reference_img[:, i_x::n_x_fold, i_y]
            temp_rho = np.matmul(np.linalg.pinv(temp_sens), temp_signal)
            unfolded_image[i_x::n_x_fold, i_y] = temp_rho

    return unfolded_image

"""
    Create paths
"""

file_path = '/media/bugger/MyBook/data/7T_scan/cardiac/2021_03_06/V9_17911'

file_list = os.listdir(file_path)
filter_on = 'cine'
trans_cine_cpx = [x for x in file_list if filter_on in x and x.endswith('.cpx')][1]
trans_cine_cpx = os.path.join(file_path, trans_cine_cpx)

filter_on = 'refscan'
refscan_cpx = [x for x in file_list if filter_on in x and x.endswith('.cpx')][0]
refscan_cpx = os.path.join(file_path, refscan_cpx)

"""
    Load data
"""

cpx_obj = read_cpx.ReadCpx(trans_cine_cpx)
cine_img = cpx_obj.get_cpx_img()
cine_par = cpx_obj.get_par_file()

cpx_obj_ref = read_cpx.ReadCpx(refscan_cpx)
ref_img = cpx_obj_ref.get_cpx_img()
ref_par = cpx_obj_ref.get_par_file()

# Extract offset in one direction....
# Taking first index.. because there should be only one key with Off Centre in it.
off_centre_ref = [np.array(v.split()).astype(float) for k, v in ref_par.items() if 'Off Centre' in k][0]
anglulation_ref = [np.array(v.split()).astype(float) for k, v in ref_par.items() if 'Angulation' in k][0]
# FOV related stuff... below is in mm
fov_ref = [np.array(v.split()).astype(float) for k, v in ref_par.items() if 'FOV' in k][0]
#
ref_z = ref_img.shape[cpx_obj_ref.sub_index_col.index('slice')]
ref_x, ref_y = ref_img.shape[-2:]
# This should now be in (ap, fh, lr) order me thinks
dim_ref = [ref_z, ref_y, ref_x]
# This should then give the voxel sizes...
voxel_ref = (fov_ref / dim_ref)

off_centre_cine = [np.array(v.split()).astype(float) for k, v in cine_par.items() if 'Off Centre' in k][0]
anglulation_cine = [np.array(v.split()).astype(float) for k, v in cine_par.items() if 'Angulation' in k][0]
# FOV related stuff... below is in mm
fov_cine = [np.array(v.split()).astype(float) for k, v in cine_par.items() if 'FOV' in k][0]

#
cine_z = cine_img.shape[cpx_obj.sub_index_col.index('slice')]
cine_y, cine_x = cine_img.shape[-2:]
# This should now be in (ap, fh, lr) order me thinks
dim_cine = [cine_y, cine_z, cine_x]
voxel_cine = (fov_cine / dim_cine)

# Distance compared to mid slice ref scan.
comp_ap, comp_fh, comp_lr = ((off_centre_ref - off_centre_cine) / (voxel_ref)).astype(int)

# 8 slices from center (that should be the offset)
# slices.... is in AP richting
# x, y ... is in LR, FH richting
# Not sure if ref_y is lr or fh... doesnt matter in this case...
coil_img = ref_img[:, 0, :, 0, 0, 0, 0, :, ref_y // 2 - comp_fh]
body_img = ref_img[0, 1, :, 0, 0, 0, 0, :, ref_y // 2 - comp_fh]
hplotc.ListPlot(body_img, augm='np.abs')
hplotc.ListPlot(np.abs(coil_img).sum(axis=0), augm='np.abs')

# Oke nu hebben we... 150 x 128 voxels voor de ref scan
# And the FOV is...   600 x 700 mm (ap x lr)
sel_time = 0
cine_time = []
n_card = cine_img.shape[cpx_obj.sub_index_col.index('hps')]
for sel_time in range(n_card):
    # Flip it because.. shit needs to be alligned.
    # And we have 178 x 620 for the other...
    # Here the FOV is... 279.31 x 450 mm (ap x lr)
    folded_img = cine_img[-8:, 0, 0, sel_time, 0, 0, 0, ::-1, ::-1]

    # This should give the amount of slices that I need to take from the reference scan
    n_ref = (fov_cine / voxel_ref)  #  .astype(int)

    min_n_ap = dim_ref[0] / 2 - n_ref[0] / 2 - comp_ap
    min_n_ap = np.round(min_n_ap).astype(int)
    max_n_ap = dim_ref[0] / 2 + n_ref[0] / 2 - comp_ap
    max_n_ap = np.round(max_n_ap).astype(int)
    min_n_lr = dim_ref[2] / 2 - n_ref[2] / 2 - comp_lr
    min_n_lr = np.round(min_n_lr).astype(int)
    max_n_lr = dim_ref[2] / 2 + n_ref[2] / 2 - comp_lr
    max_n_lr = np.round(max_n_lr).astype(int)

    # Using this body imaging produces worse results than with the one we choose
    # sel_coilimg_rel = coil_img / body_img
    sel_coilimg_rel = coil_img / np.abs(coil_img).sum(axis=0)

    hplotc.ListPlot(sel_coilimg_rel.sum(axis=0), augm='np.angle')
    hplotc.SlidingPlot(sel_coilimg_rel, augm='np.abs')
    import matplotlib.pyplot as plt
    plt.vlines(min_n_lr, min_n_ap, max_n_ap, colors='r')
    plt.vlines(max_n_lr, min_n_ap, max_n_ap, colors='r')
    plt.hlines(min_n_ap, min_n_lr, max_n_lr, colors='r')
    plt.hlines(max_n_ap, min_n_lr, max_n_lr, colors='r')

    sel_coilimg_rel = sel_coilimg_rel[:, min_n_ap-5:max_n_ap+5, min_n_lr:max_n_lr]

    sense_factor = 3
    target_size = (178 * sense_factor, 620)
    scale_factor = target_size[-1] / sel_coilimg_rel.shape[-1]
    # Resize it...
    resized_ref = [sktransform.resize(x.real, target_size, preserve_range=True) +
                   1j * sktransform.resize(x.imag, target_size, preserve_range=True) for x in sel_coilimg_rel]
    # Rescaling did not show any improvement over resizing
    resized_ref = np.array(resized_ref)
    mask_obj = hplotc.MaskCreator(resized_ref)

    # hplotc.ListPlot([resized_ref.sum(axis=0), folded_img.sum(axis=0)], augm='np.abs', aspect='auto')
    # Oke... een poging gedaan...
    test = unfold(folded_img, resized_ref[-8:]*mask_obj.mask, sense_factor)
    test = unfold(folded_img, resized_ref[-8:] , sense_factor)
    hplotc.ListPlot(test, augm='np.abs', aspect='auto')

    cine_time.append(test)

hplotc.ListPlot(np.array(cine_time)[0], augm='np.abs')