"""

Registering files is one thing... this class handles the loading and storage of directories of files

"""

import helper.plot_class as hplotc
import time
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


class RegistrationProcess:
    """
    Uses the Registration class to register objects
    """
    def __init__(self, patient_files, patient_mask_files, b1_file, dest_path, display=False,
                 registration_options='rigidaffine', n_cores=False, data_type=None, **kwargs):

        self.debug = kwargs.get('debug', False)

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
        self.b1p_array_registered = None
        self.b1m_array = np.moveaxis(mat_obj['B1minus'][0][0], -1, 0)
        self.b1m_array_registered = None
        self.b1_mask = mat_obj['Mask'][0][0]
        # Taking sum and ABS so we can display the image
        # This image is not used for the registration. The masks are
        self.b1p_visual = np.abs(self.b1p_array.sum(axis=0))

        # These are loaded in a separate function
        self.current_slice_index = None
        self.max_slice = None
        self.current_file = None
        self.current_file_name_no_ext = None
        self.current_mask_file = None
        self.patient_array = np.empty(None)
        self.patient_slice = np.empty(None)
        self.patient_mask = np.empty(None)
        self.patient_mask_slice = np.empty(None)

    def save_result(self):
        print('Storing current status of the object')

        file_name = self.b1_file_name_no_ext + '_to_' + self.patient_id + '_' + self.current_file_name_no_ext + '.h5'
        # h5 save..
        input_file_path = os.path.join(self.input_dest_path, file_name)
        target_file_path = os.path.join(self.target_dest_path, file_name)
        mask_file_path = os.path.join(self.mask_dest_path, file_name)
        target_clean_file_path = os.path.join(self.target_clean_dest_path, file_name)
        with h5py.File(input_file_path, 'w') as h5_obj:
            h5_obj.create_dataset('data', data=self.b1m_array_registered)

        with h5py.File(target_file_path, 'w') as h5_obj:
            h5_obj.create_dataset('data', data=self.b1p_array_registered)

        with h5py.File(target_clean_file_path, 'w') as h5_obj:
            h5_obj.create_dataset('data', data=self.patient_array)

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

    def resize_b1_array(self):
        self.b1_mask = sktransf.resize(self.b1_mask, self.patient_array_shape, preserve_range=True).astype(int)
        self.b1p_visual = sktransf.resize(self.b1p_visual, self.patient_array_shape, preserve_range=True)
        # Make sure that everthing is in the center...
        b1_affine_coords, b1_crop_coords = harray.get_center_transformation_coords(self.b1_mask)
        self.b1_mask = harray.apply_center_transformation(self.b1_mask, affine_coords=b1_affine_coords, crop_coords=b1_crop_coords)
        self.b1p_visual = harray.apply_center_transformation(self.b1p_visual, affine_coords=b1_affine_coords, crop_coords=b1_crop_coords)

        temp_b1p = [harray.resize_complex_array(x, new_shape=self.patient_array_shape, preserve_range=True) for x in self.b1p_array]
        temp_b1m = [harray.resize_complex_array(x, new_shape=self.patient_array_shape, preserve_range=True) for x in self.b1m_array]

        b1p_array = []
        b1m_array = []
        for x, y in zip(temp_b1p, temp_b1m):
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

    def run(self):
        # Register ALL the files from the chosen patient on the b1_file (or vice versa)
        for i_index in range(len(self.patient_files)):
            print('Running file: ', self.patient_files[i_index])
            self.run_file_mp(i_index)

    def run_file(self, file_index):
        # Register a SINGLE file from the chosen patient on the b1_file (or vice versa)
        self.set_single_file(file_index)

        results = []
        for i_slice in range(self.max_slice):
            print(f'Processing slice {i_slice} / {self.max_slice}', end='\r')
            b1p_registered, b1m_registered, patient_array, mask_array = self.register_slice(i_slice)
            if b1p_registered is not None:
                results.append((b1p_registered, b1m_registered, patient_array, mask_array))

        b1p_registered, b1m_registered, patient_array, patient_mask = zip(*results)

        self.b1p_array_registered = harray.convert_cpx2int16(np.array(b1p_registered), stack_axis=1)
        self.b1m_array_registered = harray.convert_cpx2int16(np.array(b1p_registered), stack_axis=1)

        # Needed to be redefined because of possible shifting...
        self.patient_array = np.array(patient_array)
        self.patient_mask = np.array(patient_mask).astype(int)

        self.save_result()

    def run_file_mp(self, file_index):
        # Register a SINGLE file from the chosen patient on the b1_file (or vice versa)
        self.set_single_file(file_index)

        with mp.Pool(processes=self.n_cores) as p:
            results = p.map(self.register_slice, list(range(self.max_slice)))

        b1p_registered, b1m_registered, patient_array, patient_mask = zip(*results)
        # Filter on arrays that have gone OK
        complete_index = [ii for ii, x in enumerate(b1p_registered) if x is not None]
        b1p_registered = np.array([x for ii, x in enumerate(b1p_registered) if ii in complete_index])
        b1m_registered = np.array([x for ii, x in enumerate(b1m_registered) if ii in complete_index])
        patient_array = np.array([x for ii, x in enumerate(patient_array) if ii in complete_index])
        patient_mask = np.array([x for ii, x in enumerate(patient_mask) if ii in complete_index])

        b1p_registered = harray.scale_minmax(b1p_registered, is_complex=True, axis=(-3, -2, -1))
        self.b1p_array_registered = harray.convert_cpx2int16(b1p_registered, stack_axis=1)

        b1m_registered = harray.scale_minmax(b1m_registered, is_complex=True, axis=(-3, -2, -1))
        self.b1m_array_registered = harray.convert_cpx2int16(b1m_registered, stack_axis=1)

        # Needed to be redefined because of possible shifting...
        self.patient_array = np.array(patient_array)
        self.patient_mask = np.array(patient_mask).astype(int)
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
        print('Setting single file')
        # Loads the whole array into memory
        # This was better than loading slice per slice per slice...
        if (self.current_file != self.patient_files[file_index]) or (reload is True):
            self.current_file = self.patient_files[file_index]
            self.current_file_name_no_ext = os.path.splitext(os.path.basename(self.current_file))[0]

            if self.debug:
                t0 = time.time()
                print('Start loading ', self.current_file)

            with h5py.File(self.current_file, 'r') as h5_obj:
                self.patient_array = np.array(h5_obj['data'])

            if self.debug:
                print('Done loading ', self.current_file)
                print('Took ', time.time() - t0)

            if self.debug:
                t0 = time.time()
                print('Start loading ', self.current_mask_file)

            self.current_mask_file = self.patient_mask_files[file_index]
            with h5py.File(self.current_mask_file, 'r') as h5_obj:
                self.patient_mask = np.array(h5_obj['data']).astype(int)

            if self.debug:
                print('Done loading ', self.current_mask_file)
                print('Took ', time.time() - t0)

            self.max_slice = self.patient_array.shape[0]
            self.patient_array_shape = self.patient_array.shape[-2:]
            self.resize_b1_array()
        else:
            print('Already loaded file ', self.patient_files[file_index])

    def register_slice(self, sel_slice):
        # This registers the current selected slice...
        self.patient_slice = self.patient_array[sel_slice]
        self.patient_mask_slice = self.patient_mask[sel_slice]

        temp_array, temp_mask = harray.get_center_transformation(self.patient_slice, self.patient_mask_slice)
        self.patient_slice = temp_array
        self.patient_mask_slice = temp_mask

        self.registration_obj = Registration.Registration(A=self.b1p_visual, B=self.patient_slice,
                                                          A_mask=self.b1_mask, B_mask=self.patient_mask_slice,
                                                          registration_options=self.registration_options)

        _ = self.registration_obj.register_mask()
        temp_reg_mask = self.registration_obj.apply_registration(self.b1_mask)

        temp_dice_score = hmisc.dice_metric(temp_reg_mask, self.patient_mask_slice)
        if self.debug:
            print('Current dice score ', temp_dice_score)
        if self.display:
            self.registration_obj.display_content()
            self.registration_obj.display_mask_validation()

        if temp_dice_score < 0.1:
            b1p_registered = np.array([self.registration_obj.apply_registration_cpx(x) for x in self.b1p_array])
            b1m_registered = np.array([self.registration_obj.apply_registration_cpx(x) for x in self.b1m_array])
        else:
            b1p_registered = None
            b1m_registered = None

        # Return the centered + registerd b1p and b1m maps
        # But also the centered patient slice and patient mask..
        return b1p_registered, b1m_registered, self.patient_slice, self.patient_mask_slice

    def run_debug(self, file_index, slice_index):
        self.set_single_file(file_index=file_index)
        b1p_registered, b1m_registered, self.patient_slice, temp_reg_mask = self.register_slice(sel_slice=slice_index)
        plot_obj = hplotc.ListPlot([self.b1p_array, self.b1m_array], augm='np.abs')
        plot_obj.figure.savefig(os.path.join(self.dest_path, 'b1_stuff.png'))
        plot_obj = hplotc.ListPlot([b1p_registered, b1m_registered, temp_reg_mask * (1 + self.patient_mask_slice)], augm='np.abs')
        plot_obj.figure.savefig(os.path.join(self.dest_path, 'b1_registration_stuff.png'))
        plot_obj = hplotc.ListPlot([self.patient_slice, self.patient_mask_slice, self.b1_mask], augm='np.abs')
        plot_obj.figure.savefig(os.path.join(self.dest_path, 'mask_stuff.png'))


if __name__ == "__main__":
    import numpy as np
    import helper.plot_class as hplotc
    import os
    import h5py

    dest_dir = '/local_scratch/sharreve/mri_data/registrated_h5/validation'
    reg_b1p = '/local_scratch/sharreve/mri_data/registrated_h5/validation/target/M23_to_28_MR_20210325_0006.h5'
    reg_b1m = '/local_scratch/sharreve/mri_data/registrated_h5/validation/input/M23_to_28_MR_20210325_0006.h5'
    reg_rho = '/local_scratch/sharreve/mri_data/registrated_h5/validation/target_clean/M23_to_28_MR_20210325_0006.h5'
    reg_mask = '/local_scratch/sharreve/mri_data/registrated_h5/validation/mask/M23_to_28_MR_20210325_0006.h5'
    orig_mask = '/local_scratch/sharreve/mri_data/mask_h5/28_MR/MRL/20210325_0006.h5'

    with h5py.File(reg_rho, 'r') as f:
        n_slice = f['data'].shape[0]

    sel_slice = n_slice // 2
    with h5py.File(reg_rho, 'r') as f:
        rho_array = f['data'][sel_slice]

    with h5py.File(reg_b1p, 'r') as f:
        b1p_array = f['data'][sel_slice]

    with h5py.File(reg_b1m, 'r') as f:
        b1m_array = f['data'][sel_slice]

    with h5py.File(reg_mask, 'r') as f:
        mask_array = f['data'][sel_slice]

    with h5py.File(orig_mask, 'r') as f:
        orig_mask_array = f['data']
        max_slice = orig_mask_array.shape[0]
        n_min = int(max_slice * 0.3)
        n_max = max_slice - int(max_slice * 0.3)
        orig_mask_array = orig_mask_array[n_min:n_max][sel_slice]


    # plot_obj = hplotc.ListPlot([b1p_array, b1m_array, rho_array], augm='np.abs')
    for i, i_coil in enumerate(b1p_array):
        plot_obj = hplotc.ListPlot([i_coil], augm='np.abs')
        plot_obj.figure.savefig(os.path.join(dest_dir, f'b1p_coil_{i}.png'))

    for i, i_coil in enumerate(b1m_array):
        plot_obj = hplotc.ListPlot([i_coil], augm='np.abs')
        plot_obj.figure.savefig(os.path.join(dest_dir, f'b1m_coil_{i}.png'))

    plot_obj = hplotc.ListPlot(rho_array, augm='np.abs')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'rho_array.png'))

    plot_obj = hplotc.ListPlot(mask_array, augm='np.abs')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'mask_array.png'))

    import scipy.ndimage
    plot_obj = hplotc.ListPlot(scipy.ndimage.binary_fill_holes(harray.smooth_image(mask_array, n_kernel=8)), augm='np.abs')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'mask_array_filled.png'))

    plot_obj = hplotc.ListPlot(orig_mask_array, augm='np.abs')
    plot_obj.figure.savefig(os.path.join(dest_dir, f'orig_mask_array.png'))
