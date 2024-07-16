
import itertools
import shutil
import helper.plot_class as hplotc
import h5py
import re
import helper.array_transf as harray
import numpy as np
import helper.misc as hmisc
import os
import skimage.transform as sktransf
from skimage.metrics import structural_similarity
from helper.metric import dice_score

"""
This was the thing that I finally used...
"""


class CompareImages:
    # Loads arrays.. and is able to compare them..
    # Comparisson is now done based on masked similarity
    def __init__(self, file_1, file_2, mask_file_1=None, mask_file_2=None):
        self.array_1 = self.load_array(file_1)
        self.array_2 = self.load_array(file_2)
        if mask_file_1 is not None:
            self.mask_array_1 = self.load_array(mask_file_1)
        else:
            self.mask_array_1 = None
        if mask_file_2 is not None:
            self.mask_array_2 = self.load_array(mask_file_2)
        else:
            self.mask_array_2 = None
        # Transform the image to the center...
        self.array_1, self.mask_array_1 = harray.get_center_transformation_3d(self.array_1, x_mask=self.mask_array_1)
        self.array_2, self.mask_array_2 = harray.get_center_transformation_3d(self.array_2, x_mask=self.mask_array_2)
        self.max_slice_1 = self.array_1.shape[0]
        self.max_slice_2 = self.array_2.shape[0]

    def load_array(self, input_file):
        if isinstance(input_file, str):
            temp_name, temp_ext = os.path.splitext(input_file)
            if temp_ext == '.h5':
                with h5py.File(input_file, 'r') as h5_obj:
                    input_array = np.array(h5_obj['data'])
            elif temp_ext == '.npy':
                input_array = np.array(input_file)
            else:
                input_array = None
        elif isinstance(input_file, np.ndarray):
            input_array = input_file
        elif isinstance(input_file, list):
            input_array = np.array(input_file)
        else:
            input_array = None
        return input_array
    def compare_array(self):
        max_slice_1 = self.array_1.shape[0]
        max_slice_2 = self.array_2.shape[0]
        # Which one is the minimum...
        # Select based on that...
        min_slice = min(max_slice_1, max_slice_2)
        arg_min = np.argmin([max_slice_1, max_slice_2])
        max_slice = max(max_slice_1, max_slice_2)
        coefficient = max_slice / min_slice
        distance_avg = 0
        # Misschien niet alles doen...? Scheelt tijd wellicht?
        for sel_slice_min in range(min_slice):
            sel_slice_max = int(coefficient * sel_slice_min)
            if arg_min == 0:
                # Now the minimum slice comes from the input...
                sel_slice_1 = sel_slice_min
                sel_slice_2 = sel_slice_max
            else:
                # And here it comes from the target..
                sel_slice_2 = sel_slice_min
                sel_slice_1 = sel_slice_max
            temp1 = harray.scale_minmax(self.array_1[sel_slice_1] * self.mask_array_1[sel_slice_1])
            temp2 = harray.scale_minmax(self.array_2[sel_slice_2] * self.mask_array_2[sel_slice_2])
            min_shape = min(temp1.shape[-2:], temp2.shape[-2:])
            # Turning anti aliasing on False to get better images...
            temp2 = sktransf.resize(temp2,  min_shape, preserve_range=True, anti_aliasing=False)
            temp1 = sktransf.resize(temp1, min_shape, preserve_range=True, anti_aliasing=False)
            distance = structural_similarity(temp1, temp2)
            distance_avg += distance / min_slice
        return distance_avg

    def compare_mask_array(self):
        max_slice_1 = self.array_1.shape[0]
        max_slice_2 = self.array_2.shape[0]
        # Which one is the minimum...
        # Select based on that...
        min_slice = min(max_slice_1, max_slice_2)
        arg_min = np.argmin([max_slice_1, max_slice_2])
        max_slice = max(max_slice_1, max_slice_2)
        coefficient = max_slice / min_slice
        avg_score = 0
        for sel_slice_min in range(min_slice):
            sel_slice_max = int(coefficient * sel_slice_min)
            if arg_min == 0:
                # Now the minimum slice comes from the input...
                sel_slice_1 = sel_slice_min
                sel_slice_2 = sel_slice_max
            else:
                # And here it comes from the target..
                sel_slice_2 = sel_slice_min
                sel_slice_1 = sel_slice_max
            temp1 = self.mask_array_1[sel_slice_1]
            temp2 = self.mask_array_2[sel_slice_2]
            min_shape = min(temp1.shape[-2:], temp2.shape[-2:])
            temp1 = sktransf.resize(temp1, min_shape, preserve_range=True, anti_aliasing=False).astype(int)
            temp2 = sktransf.resize(temp2,  min_shape, preserve_range=True, anti_aliasing=False).astype(int)
            # WHY DONT YOU DO THE DICE SCORE???
            dice_value = dice_score(temp1.ravel(), temp2.ravel())
            # temp1_sum_zero = (temp1 == 0).sum()
            # temp2_sum_zero = (temp2 == 0).sum()
            # distance_avg += abs(temp1_sum_zero - temp2_sum_zero) / min_slice
            avg_score += dice_value / min_slice
        return avg_score


if __name__ == "__main__":
    """
    Create paths
    """

    # These contain a variable amount of 3D scans.....
    ddata_rho = '/local_scratch/sharreve/mri_data/prostate_h5'
    dmask_rho = '/local_scratch/sharreve/mri_data/mask_h5'
    dest_dir = '/home/sharreve/local_scratch/mri_data/prostate_weighting_h5'

    hmisc.create_datagen_dir(dest_dir, type_list=('test', 'train', 'validation'),
                             data_list=('input', 'target', 'mask'))

    list_prostate_patient = []
    # List of number of prostate_mri_mrl patients
    for i_patient in os.listdir(ddata_rho):
        temp = os.path.join(ddata_rho, i_patient)
        mri_bool = ('MRI' in os.listdir(temp))
        mrl_bool = ('MRL' in os.listdir(temp))
        if mri_bool and mrl_bool:
            list_prostate_patient.append(i_patient)

    dict_split = hmisc.create_train_test_val_files(list_prostate_patient)
    data_type = 'validation'
    # data_type = 'test'
    # data_type = 'train'
    selected_prostate_list = dict_split[data_type]
    for i_patient in selected_prostate_list:
        print('Processing patient ', i_patient)
        patient_path_MRL = os.path.join(ddata_rho, i_patient, 'MRL')
        patient_path_MRI = os.path.join(ddata_rho, i_patient, 'MRI')
        # Get the number of
        patient_file_list_MRI = [os.path.join(patient_path_MRI, x) for x in os.listdir(patient_path_MRI) if x.endswith('h5') and 'transversal' in x.lower()]
        n_slices_mri = np.array([harray.get_slices_h5_file(x) for x in patient_file_list_MRI])
        patient_file_list_MRL = [os.path.join(patient_path_MRL, x) for x in os.listdir(patient_path_MRL) if x.endswith('h5') and 'transversal' in x.lower()]
        n_slices_mrl = np.array([harray.get_slices_h5_file(x) for x in patient_file_list_MRL])
        # Find which MRI files are closest to the MRL files
        n_length_difference_mri = [np.min(abs(n_slices_mrl - i_n_mri)) for i_n_mri in n_slices_mri]
        min_difference_value = np.min(n_length_difference_mri)
        index_min = np.argwhere(n_length_difference_mri == min_difference_value).reshape(-1)
        mri_closest_length_files = np.array(patient_file_list_MRI)[index_min]
        mri_closest_length_slices = np.array(n_slices_mri)[index_min] # These should all be the same anyway...
        n_length_difference_mrl = [abs(i_n_mrl - mri_closest_length_slices[0]) for i_n_mrl in n_slices_mrl]
        min_difference_value = np.min(n_length_difference_mrl)
        index_min = np.argwhere(n_length_difference_mrl == min_difference_value).reshape(-1)
        mrl_closest_length_files = np.array(patient_file_list_MRL)[index_min]
        mrl_closest_length_slices = np.array(n_slices_mrl)[index_min]

        cart_product_files = list(itertools.product(mri_closest_length_files, mrl_closest_length_files))
        avg_dice_score_files = []
        icounter = 0
        for i_mri, i_mrl in cart_product_files:
            print(icounter, " / ", len(cart_product_files), end='\r')
            base_name, _ = os.path.splitext(os.path.basename(i_mrl))
            i_mrl_file_mask = re.sub('prostate_h5', 'mask_h5', i_mrl)
            i_mri_file_mask = re.sub('prostate_h5', 'mask_h5', i_mri)
            compare_obj = CompareImages(i_mri, i_mrl, mask_file_1=i_mri_file_mask, mask_file_2=i_mrl_file_mask)
            compare_score = compare_obj.compare_mask_array()
            avg_dice_score_files.append(compare_score)
            icounter += 1

        best_score_index = np.argmax(avg_dice_score_files)
        selected_mri_file, selected_mrl_file = cart_product_files[best_score_index]
        selected_mrl_file_mask = re.sub('prostate_h5', 'mask_h5', selected_mrl_file)
        selected_mri_file_mask = re.sub('prostate_h5', 'mask_h5', selected_mri_file)
        # Destination file locations....
        # This is the INPUT. It should be the 1.5T file (MRL)
        file_name = i_patient + '.h5'
        file_name_input_mask = i_patient + '_input.h5'
        dest_input = os.path.join(dest_dir, data_type, 'input', file_name)
        dest_input_mask = os.path.join(dest_dir, data_type, 'mask', file_name_input_mask)
        # This is the TARGET. It should be the 3T file (MRI)
        file_name_target_mask = i_patient + '_target.h5'
        dest_target = os.path.join(dest_dir, data_type, 'target', file_name)
        dest_target_mask = os.path.join(dest_dir, data_type, 'mask', file_name_target_mask)
        shutil.copy(selected_mrl_file, dest_input)
        shutil.copy(selected_mrl_file_mask, dest_input_mask)
        shutil.copy(selected_mri_file, dest_target)
        shutil.copy(selected_mri_file_mask, dest_target_mask)

