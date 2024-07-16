import helper.array_transf as harray
import numpy as np
import os
import scipy.io
import helper.plot_class as hplotc
import reconstruction.ReadCpx as read_cpx

"""
Unfolding is annoying. Cant find the right view...
"""

ddata = '/media/bugger/WORK_USB/rcget/survey_data'
ddata_sense = '/media/bugger/WORK_USB/rcget'
ddata_image = os.path.join(ddata, 'survey_data.mat')
ddata_image_T1 = os.path.join(ddata, 't1_data.mat')
ddata_coil_survey = os.path.join(ddata_sense, '20211001_092218_CoilSurveyScan.cpx')
ddata_sense_ref = os.path.join(ddata_sense, '20211001_092229_SenseRefScan.cpx')

# Read the coil survey..
coil_survey_cpx = read_cpx.ReadCpx(ddata_coil_survey)
coil_survey_img = coil_survey_cpx.get_cpx_img()
print('Order of columns ', coil_survey_cpx.sub_index_col)
# hplotc.SlidingPlot(np.abs(coil_survey_img).sum(axis=0), augm='np.abs')

# Read the SENSE data..
sense_cpx = read_cpx.ReadCpx(ddata_sense_ref)
sense_img = sense_cpx.get_cpx_img()
print('Order of columns ', sense_cpx.sub_index_col)
# hplotc.SlidingPlot(np.abs(sense_img).sum(axis=0), augm='np.abs')
# hplotc.SlidingPlot(np.moveaxis(np.squeeze(np.abs(sense_img).sum(axis=0)[0]), -1, 0), augm='np.abs')
# hplotc.SlidingPlot(np.abs(sense_img), augm='np.abs')

body_coil_image = np.squeeze(sense_img[0, 1])
coil_array_image = np.squeeze(sense_img[:, 0])
# hplotc.SlidingPlot(body_coil_image)
# hplotc.SlidingPlot(coil_array_image)

# Read the folded survey data
mat_obj = scipy.io.loadmat(ddata_image)
A = np.moveaxis(mat_obj['survey_data'][0][0], (0, 1), (-2, -1))
A = np.squeeze(A)

# Read the folded T1 data...
# mat_obj = scipy.io.loadmat(ddata_image_T1)
# A = np.moveaxis(mat_obj['t1_data'][0][0], (0, 1), (-2, -1))
# A = np.squeeze(A)

hplotc.SlidingPlot(A.sum(axis=1))

# Set reference image...
reference_img = coil_array_image / (np.abs(body_coil_image) ** 0.5)[None]
_, nz, ny, nx = reference_img.shape
reference_img = reference_img[:, :, :, nx//2]
reference_img = harray.resize_complex_array(reference_img, (32, 320, 128))
hplotc.SlidingPlot(reference_img)
# n_coils = folded_image.shape[0]
n_coils = 16
reference_img = reference_img[:n_coils]

for sel_loc in [1]:
    A_loc = A[:, sel_loc]
    A_coil_sum = A_loc.sum(axis=(-2, -1))
    import matplotlib.pyplot as plt
    sel_index_coil = np.argwhere(A_coil_sum != 0).ravel()
    A_coil_sel = A_loc[sel_index_coil]
    folded_image = A_coil_sel[:n_coils]
    unfolded_image = np.zeros(reference_img.shape[-2:], dtype=complex)
    n_c, n_x, n_y = reference_img.shape
    folding_factor = 2
    # folding_dimension = 'y'
    folding_dimension = 'x'
    if folding_dimension == 'y':
        n_y_fold = int(n_y / folding_factor)
        for i_x in range(n_x):
            for i_y in range(n_y_fold):
                temp_signal = folded_image[:, i_x, i_y]
                temp_sens = reference_img[:, i_x, i_y::n_y_fold]
                temp_rho = np.matmul(np.linalg.pinv(temp_sens), temp_signal)
                unfolded_image[i_x, i_y::n_y_fold] = temp_rho
    else:
        n_x_fold = int(n_x / folding_factor)
        for i_x in range(n_x_fold):
            for i_y in range(n_y):
                temp_signal = folded_image[:, i_x, i_y]
                temp_sens = reference_img[:, i_x::n_x_fold, i_y]
                temp_rho = np.matmul(np.linalg.pinv(temp_sens), temp_signal)
                unfolded_image[i_x::n_x_fold, i_y] = temp_rho

    hplotc.ListPlot(unfolded_image, augm='np.abs', title=sel_loc)