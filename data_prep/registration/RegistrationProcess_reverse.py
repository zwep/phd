"""

Registering files is one thing... this class handles the loading and storage of directories of files

"""

import helper.misc as hmisc
import os
import time
import skimage.transform as sktransf
import helper.array_transf as harray
import scipy.io
import h5py
import numpy as np
import data_prep.registration.Registration as Registration
import multiprocessing as mp
import re


class RegistrationProcessReverse:
    def __init__(self, patient_files, patient_mask_files, b1_file, dest_path, display=False,
                 registration_options='rigidaffine', n_cores=False, data_type=None):

        # Defining paths....
        # Datatype should be validation, train, test...
        dest_base_path = os.path.join(dest_path, data_type)
        self.dest_path = dest_path
        self.input_dest_path = os.path.join(dest_base_path, 'input')
        self.target_dest_path = os.path.join(dest_base_path, 'target')
        self.target_clean_dest_path = os.path.join(dest_base_path, 'target_clean')
        self.mask_dest_path = os.path.join(dest_base_path, 'mask')

        self.display = display

        self.patient_id = re.findall("([0-9]+_MR)", patient_files[0])[0]
        self.patient_files = patient_files
        self.patient_mask_files = patient_mask_files
        self.b1_file = b1_file

        self.registration_options = registration_options
        self.registration_obj = None

        self.n_cores = n_cores

        # Load the b1p file
        mat_obj = scipy.io.loadmat(b1_file)['Model']
        self.b1_file_name_no_ext = os.path.splitext(os.path.basename(b1_file))[0]
        # The channels are on the last axis. Move them to the front
        self.b1p_array = np.moveaxis(mat_obj['B1plus'][0][0], -1, 0)
        self.b1m_array = np.moveaxis(mat_obj['B1minus'][0][0], -1, 0)
        self.b1_mask = mat_obj['Mask'][0][0]
        self.b1_shape = self.b1m_array.shape[-2:]
        # Taking sum and ABS so we can display the image
        # This image is not used for the registration. The masks are
        self.b1p_visual = np.abs(self.b1p_array.sum(axis=0))
        self.center_b1_file()

        # These are loaded in a separate function
        self.current_slice_index = None
        self.max_slice = None
        self.current_file = None
        self.current_file_name_no_ext = None
        self.current_mask_file = None
        self.patient_array = np.empty(None)
        self.patient_array_registered = np.empty(None)
        self.patient_slice = np.empty(None)
        self.patient_mask = np.empty(None)
        self.patient_mask_registered = np.empty(None)
        self.patient_mask_slice = np.empty(None)

    def center_b1_file(self):
        # Make sure that everthing is in the center...
        b1_affine_coords, b1_crop_coords = harray.get_center_transformation_coords(self.b1_mask)
        self.b1_mask = harray.apply_center_transformation(self.b1_mask, affine_coords=b1_affine_coords,
                                                          crop_coords=b1_crop_coords)
        self.b1p_visual = harray.apply_center_transformation(self.b1p_visual, affine_coords=b1_affine_coords,
                                                             crop_coords=b1_crop_coords)
        b1p_array = []
        b1m_array = []
        for x, y in zip(self.b1p_array, self.b1m_array):
            temp_x = harray.apply_center_transformation(x,
                                                        affine_coords=b1_affine_coords,
                                                        crop_coords=b1_crop_coords,
                                                        dtype=np.complex64)
            temp_y = harray.apply_center_transformation(y,
                                                        affine_coords=b1_affine_coords,
                                                        crop_coords=b1_crop_coords,
                                                        dtype=np.complex64)
            b1p_array.append(temp_x)
            b1m_array.append(temp_y)

        self.b1p_array = np.array(b1p_array)
        self.b1m_array = np.array(b1m_array)

    def save_result(self):
        print('Storing current status of the object')

        file_name = self.patient_id + '_to_' + self.b1_file_name_no_ext + '_' + self.current_file_name_no_ext + '.h5'
        # h5 save..
        input_file_path = os.path.join(self.input_dest_path, self.b1_file_name_no_ext + '.h5')
        target_file_path = os.path.join(self.target_dest_path, self.b1_file_name_no_ext + '.h5')
        mask_file_path = os.path.join(self.mask_dest_path, file_name)
        target_clean_file_path = os.path.join(self.target_clean_dest_path, file_name)

        with h5py.File(input_file_path, 'w') as h5_obj:
            h5_obj.create_dataset('data', data=self.b1m_array.astype(np.complex64))

        with h5py.File(target_file_path, 'w') as h5_obj:
            h5_obj.create_dataset('data', data=self.b1p_array.astype(np.complex64))

        with h5py.File(target_clean_file_path, 'w') as h5_obj:
            h5_obj.create_dataset('data', data=self.patient_array.astype(np.int))

        with h5py.File(mask_file_path, 'w') as h5_obj:
            h5_obj.create_dataset('data', data=self.patient_mask.astype(np.bool))

        print('Done writing')

    def get_current_status(self):
        print('Current patient id: ', self.patient_id)

        print('\n\nCurrent b1 file ',  self.b1_file)
        print('Current patient file ', self.current_file)
        print('Current patient mask file ', self.current_mask_file)

        print('\n\nStorage input data ', self.input_dest_path)
        print('Storage target data  ', self.target_dest_path)
        print('Storage target clean data  ', self.target_clean_dest_path)
        print('Storage mask data  ', self.mask_dest_path)

        print('\n\nCurrent loaded patient array ', self.patient_array.shape)
        print('Current loaded patient mask array ', self.patient_mask.shape)
        print('Current loaded patient slice ', self.patient_slice.shape)
        print('Current loaded patient mask slice ', self.patient_mask_slice.shape)

        print('\n\nCurrent loaded b1p array ', self.b1p_array.shape)
        print('Current loaded b1m array ', self.b1m_array.shape)
        print('Current loaded b1 mask array ', self.b1_mask.shape)

    def run(self):
        # Register ALL the files from the chosen patient on the b1_file (or vice versa)
        for i_index in range(len(self.patient_files)):
            print('Running file: ', self.patient_files[i_index])
            self.run_file_mp(i_index)

    def run_file_mp(self, file_index):
        # Register a SINGLE file from the chosen patient on the b1_file (or vice versa)
        self.set_single_file(file_index)

        # Take fewer slices than we actually load...
        # This is to reduce the memory footprint
        # Weve already selected the files with the MOST slices
        n_min = int(self.max_slice * 0.45)
        n_max = self.max_slice - int(self.max_slice * 0.45)

        with mp.Pool(processes=self.n_cores) as p:
            results = p.map(self.register_slice, list(range(n_min, n_max)))

        patient_array, patient_mask = zip(*results)
        # Needed to be redefined because of possible shifting...
        self.patient_array = np.array(patient_array)
        self.patient_mask = (np.array(patient_mask) > 0.1).astype(np.int)
        self.save_result()

    def run_slice(self, file_index, slice_index):
        # Register a SINGLE SLICE from a file from the chosen patient on the b1_file (or vice versa)
        self.set_single_file(file_index)
        container = self.register_slice(slice_index)
        return container

    def run_current_slice(self):
        # Simple helper function...
        container = self.register_slice(self.current_slice_index)
        return container

    def set_single_file(self, file_index, reload=False):
        # Loads the whole array into memory
        if (self.current_file != self.patient_files[file_index]) or (reload is True):
            self.current_file = self.patient_files[file_index]
            self.current_file_name_no_ext = os.path.splitext(os.path.basename(self.current_file))[0]

            with h5py.File(self.current_file, 'r') as h5_obj:
                self.patient_array = np.array(h5_obj['data'])

            self.current_mask_file = self.patient_mask_files[file_index]
            with h5py.File(self.current_mask_file, 'r') as h5_obj:
                self.patient_mask = np.array(h5_obj['data']).astype(int)

            self.max_slice = self.patient_array.shape[0]
            self.patient_array_shape = self.patient_array.shape[-2:]
            self.patient_array = sktransf.resize(self.patient_array, (self.max_slice,) + self.b1_shape, preserve_range=True)
            self.patient_mask = sktransf.resize(self.patient_mask, (self.max_slice,) + self.b1_shape, preserve_range=True)
        else:
            print('Already loaded file ', self.patient_files[file_index])

    def register_slice(self, sel_slice):
        # This registers the current selected slice...
        self.patient_slice = self.patient_array[sel_slice]
        self.patient_mask_slice = self.patient_mask[sel_slice]

        temp_array, temp_mask = harray.get_center_transformation(self.patient_slice, self.patient_mask_slice)
        self.patient_slice = temp_array
        self.patient_mask_slice = temp_mask

        self.registration_obj = Registration.Registration(A=self.patient_slice, B=self.b1p_visual,
                                                          A_mask=self.patient_mask_slice, B_mask=self.b1_mask,
                                                          registration_options=self.registration_options)

        _ = self.registration_obj.register_mask()
        temp_reg_mask = self.registration_obj.apply_registration(self.patient_mask_slice)

        temp_dice_score = hmisc.dice_metric(temp_reg_mask, self.b1_mask)
        print('Current dice score ', temp_dice_score)

        if self.display:
            self.registration_obj.display_content()
            self.registration_obj.display_mask_validation()

        patient_slice_reg = self.registration_obj.apply_registration(self.patient_slice)

        return patient_slice_reg, temp_reg_mask

