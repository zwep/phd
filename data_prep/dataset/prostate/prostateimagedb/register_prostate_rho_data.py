
"""
This was a start for the registration of prostate_mri_mrl data


"""
import pydicom
import os
import numpy as np
import scipy.io
import helper.plot_class as hplotc
import helper.array_transf as harray
import data_prep.registration.Registration as Registration

# This one is not split into test/train/val yet...
# prostate_dir = "/home/bugger/Documents/data/prostateimagedatabase/rho"
prostate_dir = "/home/bugger/Documents/data/1.5T/prostate"
# This one is already divided into train/test/val split...
b1_dir = "/media/bugger/MyBook/data/simulated/b1p_b1m_flavio_mat"

# Lets just test the class first...
prostate_file_list = [x for x in os.listdir(prostate_dir) if x.endswith('.dcm')]
b1_file_list = [x for x in os.listdir(b1_dir) if x.endswith('.mat')]

n_prostate_files = len(prostate_file_list)
n_b1_files = len(b1_file_list)
dice_score_matrix = np.zeros((n_prostate_files, n_b1_files))

i_counter = 0
for i_counter in range(10):
    sel_file = prostate_file_list[i_counter]
    prostate_file = os.path.join(prostate_dir, sel_file)

    sel_file = b1_file_list[i_counter]
    b1_file = os.path.join(b1_dir, sel_file)

    # Load B1 files from flavio
    b1_dict = scipy.io.loadmat(b1_file)['Model']
    b1_min_array = b1_dict['B1minus'][0][0]
    b1_plus_array = b1_dict['B1plus'][0][0]
    b1_plus_array = np.abs(b1_plus_array).sum(axis=-1)
    b1_mask = b1_dict['Mask'][0][0]

    # Read Prostte rho files
    if prostate_file.endswith('dcm'):
        prostate_array = pydicom.read_file(prostate_file).pixel_array
    elif prostate_file.endswith('nrrd'):
        # prostate_array, _ = nrrd.read(prostate_file)
        sel_slice = 0
        prostate_array = prostate_array[:, :, sel_slice]
        prostate_array = np.rot90(prostate_array, k=3)

    mask_obj = hplotc.MaskCreator(prostate_array[None])
    A_mask = mask_obj.mask.astype(int)
    import skimage.transform as sktransf
    prostate_array = sktransf.resize(prostate_array, b1_mask.shape, preserve_range=True)
    A_mask = sktransf.resize(A_mask, b1_mask.shape, preserve_range=True)

    registration_obj = Registration.Registration(prostate_array, b1_plus_array, A_mask=A_mask, B_mask=b1_mask,
                                                 registration_options='rigid')

    registration_obj.display_content()
    _ = registration_obj.register_mask()
    fig_handle = registration_obj.display_mask_validation()

    temp_A, temp_A_mask = harray.get_center_transformation(registration_obj.A, registration_obj.A_mask)
    registration_obj.A = temp_A
    registration_obj.A_mask = temp_A_mask
    temp_B, temp_B_mask = harray.get_center_transformation(registration_obj.B, registration_obj.B_mask)
    registration_obj.B = temp_B
    registration_obj.B_mask = temp_B_mask

    registration_obj.display_content()

    _ = registration_obj.register_mask()
    _, dice_score = registration_obj.validate_mask_mapping()
    print('Dice score ', dice_score)
    fig_handle = registration_obj.display_mask_validation()
    fig_handle.suptitle(dice_score)

    res = registration_obj.apply_registration(prostate_array)
    B_mask_approx, dice_score = registration_obj.validate_mask_mapping()
    hplotc.ListPlot([prostate_array, res * temp_B_mask])