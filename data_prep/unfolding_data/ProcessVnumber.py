import warnings
import helper.plot_class as hplotc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import scipy.ndimage
import time
import importlib
import os
import warnings
import pandas as pd
import numpy as np
import reconstruction.ReadCpx as read_cpx
import reconstruction.SenseUnfold as sense_unfold
import helper.array_transf as harray
from pandas_ods_reader import read_ods
import re
from PIL import Image

"""
Here we are going to create a class that can process the content of one Vnumber...

A Vnumber is a patient/volunteer-number in our hospital. I dont know if these things are generic across all hospitals.

"""


class ProcessVnumber:
    unfolding_classes = ['transverse', 'p2ch', 'sa', '4ch']

    def __init__(self, v_number, scan_dir, target_dir, extension_filter='cpx', save_format=None,
                 sel_number_coils=24, sense_factor=3, debug=False, status=False):

        acq_files = self.get_files_vnumber(path=scan_dir, v_number=v_number, extension_filter=extension_filter)
        self.v_number = v_number
        self.ref_file_str = [(i, x) for i, x in enumerate(acq_files) if 'sense' in x.lower()]
        self.cine_file_str = [(i, x) for i, x in enumerate(acq_files) if ('ref' not in x.lower()) and ('radial' not in x)]
        self.target_main_dir = target_dir
        self.save_format = save_format  # Can be either jpeg, or npy..
        self.sel_number_coils = sel_number_coils
        self.sense_factor = sense_factor
        # Need to implement something for this still...
        self.orientation_table = self.get_manual_orientations()

        # These below can be set in set_folded_obj()
        self.folded_obj = None
        self.folded_file_name = None
        self.folded_name = None
        # This parameter is used to denote the target sub directory (p2ch, 4ch, ..)
        # It is also used to skip over files we dont want to process (= None)
        self.target_sub_dir = None
        self.folded_param = None
        self.folded_img = None
        self.target_size = None
        self.folded_colname = None
        self.orientation_dict = None

        # Just maybe... load the refscan on the start...?
        self.sense_obj = None
        self.refscan_file_name = None
        self.mask_obj = None
        self.current_plane_acq = None

        # And set this one during stuff...
        self.refscan_plane = None
        self.mask_plane = None

        self.status = status
        self.debug = debug

        if self.status:
            print('Object initiated. Following files are present')
            print('Scan files')
            for i_file in self.cine_file_str:
                print(i_file)
            print('Refscan files')
            for i_file in self.ref_file_str:
                print(i_file)

    def set_folded_obj(self, sel_index=None, sel_file_name=None):
        if self.status:
            print('\nSetting parameters of folded object')

        # Here we set multiple objects needed for the unfolding process
        sel_index, sel_file = self.get_selected_file(sel_index=sel_index, sel_file_name=sel_file_name)

        if self.debug:
            print('\t\tDebug: sel folded obj')
            print('Obtained sel file from get selected file ', sel_file)

        base_name = os.path.basename(sel_file)
        base_name_no_ext = os.path.splitext(base_name)[0]
        # Check what kind of rotation we need for this file...
        index_rotation_tbl = [i for i, x in enumerate(self.orientation_table['file name']) if base_name_no_ext in x]
        if len(index_rotation_tbl) == 1:
            table_index = index_rotation_tbl[0]
            self.orientation_dict = dict(self.orientation_table.iloc[table_index])
        else:
            print('We have not found this file...', sel_file)
            self.orientation_dict = {'file name': sel_file,
                                     'Rotation': -1,
                                     'Remove': None,
                                     'Orientation (0, 1, 2, 3)': None}

        cpx_obj_acq, target_sub_dir = self.load_scan_obj(sel_file)
        if target_sub_dir is None:
            print("** ")
            return 'Remove'

        acq_param, acq_img, target_size = self.load_folded_file_parameters(cpx_obj_acq)

        self.folded_obj = cpx_obj_acq
        self.folded_file_name = sel_file
        self.folded_name = base_name
        self.folded_name_no_ext = base_name_no_ext
        self.target_sub_dir = target_sub_dir

        self.folded_param = acq_param
        self.folded_img = acq_img
        self.target_size = target_size
        self.folded_colname = self.folded_obj.sub_index_col
        self.prep_direction = self.folded_param['Preparation direction']

        if self.status:
            print('Loaded parameters')
            print('Loaded file name ', sel_file)
            print('Target sub-directory ', target_sub_dir)
            print('Folded image shape ', acq_img.shape)
            print('Target image shape ', target_size)
            print('Order of columns from acq object ', self.folded_colname)
            print('Preperation direction ', self.prep_direction)

        # Maybe dont add this yet...
        if self.target_sub_dir is None:
            self.orientation_dict['Remove'] = 'Yes'

        return self.orientation_dict['Remove']

    def set_slice_refscan_mask(self):
        if self.status:
            print('\nSetting new slice refscan and mask')

        assert self.folded_param is not None
        assert self.target_size is not None

        ref_acq_plane = self.sense_obj.get_refscan_acq(acq_param=self.folded_param, target_size=self.target_size)
        index_nan = np.isnan(ref_acq_plane)
        if index_nan.sum():
            ref_acq_plane[index_nan] = 0

        # This part is commented out, because masking theference scan poses too many errors and instabilities.
        # print('Shape of mask refscan before getting the slice.. ', self.mask_obj.refscan.shape)
        # mask_acq_plane = self.mask_obj.get_refscan_acq(acq_param=self.folded_param, target_size=self.target_size)
        # index_nan = np.isnan(mask_acq_plane)
        # if index_nan.sum():
        #     mask_acq_plane[index_nan] = 0

        self.refscan_plane = ref_acq_plane
        self.mask_plane = np.array([0])
        self.current_plane_acq = self.folded_file_name

        if self.status:
            print('Loaded parameters')
            print('Obtained refscan plane shape ', self.refscan_plane.shape)
            print('Obtained mask plane shape ', self.mask_plane.shape)
            print('Current planes belong to acq file ', self.current_plane_acq)

    def set_sense_object(self, cpx_obj):
        if self.status:
            print('\n Setting new SENSE object')
        sense_obj = sense_unfold.SenseUnfold(cpx_obj, n_coils=self.sel_number_coils)

        # Masking is turned off because it raised too many instabilities. Manual masking was too labor intense
        # refscan_np_ext = os.path.splitext(cpx_obj.input_file)[0] + '.npy'
        # print('Loading... ', refscan_np_ext)
        # mask_array = np.load(refscan_np_ext)
        # if os.path.isfile(refscan_np_ext):
        #     mask_obj = sense_unfold.SenseUnfold(cpx_obj, n_coils=self.sel_number_coils, pre_loaded_image=mask_array)
        #     # Add new axes..... to mimic a location...
        #     mask_obj.refscan = mask_obj.refscan[None]
        #     print('Loaded shape of refscan is ... ', mask_obj.refscan.shape)
        # else:
        #     print('Uh oh, we have not created a mask for this array yet...', mask_array)

        self.refscan_file_name = cpx_obj.input_file
        self.sense_obj = sense_obj
        # self.mask_obj = mask_obj

        if self.status:
            print('Loaded parameters')
            print('Loaded refscan file name ', self.refscan_file_name)

    def rotate_unfolded_image(self, image):
        # Uses current orientation dictionary
        if self.orientation_dict['Rotation'] is not None:
            if self.status:
                print('Rotating the unfolded image')

            if self.orientation_dict['Rotation'] == 90:
                image = np.rot90(image, axes=(-2, -1))
            elif self.orientation_dict['Rotation'] == 180:
                image = np.rot90(image, k=2, axes=(-2, -1))
            elif self.orientation_dict['Rotation'] == -180:
                image = np.rot90(image, k=-2, axes=(-2, -1))
            elif self.orientation_dict['Rotation'] == -90:
                image = np.rot90(image, k=-1, axes=(-2, -1))

        return image

    def check_update_scanfile(self, scan_file):
        # Check if we need to load it again.
        if self.folded_file_name is None:
            remove_indicator = self.set_folded_obj(sel_file_name=scan_file)
        # This is split up because we might want to process this part differently
        elif self.folded_file_name != scan_file:
            remove_indicator = self.set_folded_obj(sel_file_name=scan_file)
        else:
            remove_indicator = None
            if self.status:
                print('Current folded file is already loaded in ', self.folded_file_name)

        return remove_indicator

    def check_update_refscan(self, refscan_file, scan_file):
        # Check if we need a different refscan file
        if self.refscan_file_name != refscan_file:
            cpx_obj_acq, _ = self.load_scan_obj(refscan_file)
            # Set the newly loaded refscan object
            self.set_sense_object(cpx_obj_acq)
            # Calculate the current acq plane in the refscan...
            self.set_slice_refscan_mask()
        else:
            # If the current plane is not the same as the one we just loaded...
            # Update this one...
            if self.current_plane_acq != scan_file:
                self.set_slice_refscan_mask()

            if self.status:
                print('Current refscan file is already loaded in ', self.refscan_file_name)

    def run_single_slice(self, sel_file_index=None, sel_file_name=None):

        sel_index, sel_file = self.get_selected_file(sel_index=sel_file_index, sel_file_name=sel_file_name)

        if self.debug:
            print('\t\tDebug: run_single_slice')
            print('Obtained sel file from get selected file ', sel_file)
            print('With index :', sel_index)

        remove_indicator = self.check_update_scanfile(sel_file)

        # If we wont use it, send and early return signal
        if remove_indicator is not None:
            print('We are not going to use this file ', sel_file)
            return -1

        # Check the associated reference scan file
        filter_refscan = [x for i, x in self.ref_file_str if sel_index > i]

        if self.debug:
            print('\t\tDebug: run_single_slice')
            print('Selection of filtered rescan files ', filter_refscan)

        # Check if the loaded file even needs a refscan...
        if self.target_sub_dir in self.unfolding_classes:
            sel_refscan = filter_refscan[-1]
            self.check_update_refscan(sel_refscan, sel_file)

            # Take everything but the last two...
            temp = self.folded_img
            for _ in range(temp.ndim - 3):
                temp = np.take(temp, 0, axis=1)

            if self.debug:
                print('\t\tDebugDebug: run_single_slice')
                print('Shape of rescan_plane ', self.refscan_plane[0].shape)
                print('Shape of mask_plane ', self.mask_plane[0].shape)
                print('Shape of img', temp.shape)

            unfolded_image = self.unfold_single_slice(temp, self.refscan_plane[0], self.mask_plane[0])
            unfolded_image = self.rotate_unfolded_image(unfolded_image)
        else:
            if self.status:
                print('Current file needs no unfolding ', self.folded_file_name)

            unfolded_image = self.folded_img

        if self.save_format == 'jpeg':
            self.save_jpeg(unfolded_image)
        elif self.save_format == 'npy':
            self.save_image(unfolded_image)
        else:
            print('We are not saving the image')
        return unfolded_image

    def run_single_file(self, sel_index=None, sel_file_name=None):
        # Get the file that is requested
        sel_index, sel_file = self.get_selected_file(sel_index=sel_index, sel_file_name=sel_file_name)

        remove_indicator = self.check_update_scanfile(sel_file)
        # If we wont use it, send and early return signal
        if remove_indicator is not None:
            print('We are not going to use this file ', sel_file)
            return -1

        # Check the associated reference scan file
        filter_refscan = [x for i, x in self.ref_file_str if sel_index > i]
        # Check if the loaded file even needs a refscan...
        if self.target_sub_dir in self.unfolding_classes:
            sel_refscan = filter_refscan[-1]
            self.check_update_refscan(sel_refscan, sel_file)
            unfolded_image = self.unfold_single_file()
            unfolded_image = self.rotate_unfolded_image(unfolded_image)
        else:
            unfolded_image = self.folded_img

        if self.save_format == 'jpeg':
            self.save_jpeg(unfolded_image)
        elif self.save_format == 'npy':
            self.save_image(unfolded_image)
        else:
            print('We are not saving the image')
        return unfolded_image

    def run(self):
        print('Running the set V-number files..')
        for i, sel_file_name in self.cine_file_str:
            print('Processing file: ', sel_file_name)
            # Get the file that is requested
            sel_index, sel_file = self.get_selected_file(sel_file_name=sel_file_name)

            remove_indicator = self.check_update_scanfile(sel_file)

            if remove_indicator is not None:
                print('We are not going to use this file ', sel_file)
                continue

            # Check the associated reference scan file
            filter_refscan = [x for i, x in self.ref_file_str if sel_index > i]
            # Check if the loaded file even needs a refscan...
            if self.target_sub_dir in self.unfolding_classes:
                sel_refscan = filter_refscan[-1]
                self.check_update_refscan(sel_refscan, sel_file)
                unfolded_image = self.unfold_single_file()
                unfolded_image = self.rotate_unfolded_image(unfolded_image)
            else:
                unfolded_image = self.folded_img

            if self.save_format == 'jpeg':
                self.save_jpeg(unfolded_image)
            elif self.save_format == 'npy':
                self.save_image(unfolded_image)
            else:
                print('We are not saving the image')

            print('Shape of result...', unfolded_image.shape)

    def save_image(self, img_array):
        temp_dir = os.path.join(self.target_main_dir, self.v_number, self.target_sub_dir)
        file_dir = os.path.join(temp_dir, self.folded_name_no_ext)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)

        if os.path.isfile(file_dir):
            print('File is already there... overwriting now.')

        np.save(file_dir, img_array)

    def save_jpeg(self, img_array):
        temp_dir = os.path.join(self.target_main_dir, self.v_number, self.target_sub_dir)
        file_dir = os.path.join(temp_dir, self.folded_name_no_ext)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)

        if os.path.isfile(file_dir):
            print('File is already there... overwriting now.')

        # Take everything but the last two...
        temp = img_array
        for _ in range(temp.ndim - 2):
            temp = np.take(temp, 0, axis=0)

        single_img = np.abs(temp)
        single_img = (harray.scale_minmax(single_img) * 256).astype(np.uint8)

        im = Image.fromarray(single_img)
        im.save(file_dir + ".jpeg")

    def unfold_single_file(self):
        # Something like this can be implemented here to skip already processed files...
        # if os.path.isfile(target_file_path + '.npy'):
        #     print('==>Passing file ', target_file_path)
        #     return None
        if self.status:
            print('\n Unfolding single file')

        assert self.folded_param is not None

        ind_loc = self.folded_colname.index('loc')
        ind_card = self.folded_colname.index('hps')
        ind_slice = self.folded_colname.index('slice')
        ind_dyn = self.folded_colname.index('dyn')
        ind_echo = self.folded_colname.index('echo')
        ind_mix = self.folded_colname.index('echo')

        n_card = self.folded_img.shape[ind_card]
        n_loc = self.folded_img.shape[ind_loc]
        n_slice = self.folded_img.shape[ind_slice]
        n_dyn = self.folded_img.shape[ind_dyn]
        n_echo = self.folded_img.shape[ind_echo]
        n_mix = self.folded_img.shape[ind_mix]

        # Concatenate these tuples...
        # Remove first (=n_coils) and last two (=n_x, n_y) dimensions
        target_size = self.folded_img.shape[1:-2] + self.target_size
        # Shape should be equal to acq img.. but without the folded dimension
        cine_array = np.empty(target_size, dtype=np.complex)
        if self.status:
            print('Creating array of size ', target_size)

        for sel_loc in range(n_loc):
            ref_scan_loc = self.refscan_plane[sel_loc]
            mask_scan_loc = self.mask_plane[0]
            for sel_slice in range(n_slice):
                for sel_card in range(n_card):
                    for sel_echo in range(n_echo):
                        for sel_dyn in range(n_dyn):
                            for sel_mix in range(n_mix):
                                folded_img_card = self.folded_img[:, sel_loc, sel_slice, sel_card, sel_echo, sel_dyn, sel_mix]
                                if self.debug:
                                    print('\t\t Debug: unfold_sinlge_file')
                                    print('Current position ', sel_loc, sel_slice, sel_card, sel_echo, sel_dyn)
                                    print('Folded image shape ', folded_img_card.shape)
                                    print('Refscan shape ', ref_scan_loc.shape)
                                    print('Mask shape ', mask_scan_loc.shape)

                                    # print('Unfolded image shape ', unfolded_img.shape)

                                unfolded_img = self.unfold_single_slice(folded_img_card, ref_scan_loc, mask_scan_loc)

                                cine_array[sel_loc, sel_slice, sel_card, sel_echo, sel_dyn, sel_mix] = unfolded_img
        return cine_array

    def unfold_single_slice(self, folded_img, refscan, mask):
        # The orientation of the reference scan is very important
        # Here we have a manually set the correct orientation for the chosen scans
        ref_obj_orientation = None
        print(f'\t\t Unfolding {self.target_sub_dir} image', end='\r')
        if self.target_sub_dir == 'transverse':
            # ref_obj_orientation = refscan[:, ::-1, ::-1] * mask[:, ::-1, ::-1]
            ref_obj_orientation = refscan[:, ::-1, ::-1]
        elif self.target_sub_dir == 'p2ch':
            # ref_obj_orientation = refscan * mask
            ref_obj_orientation = refscan
        elif self.target_sub_dir == '4ch':
            if self.prep_direction == 'RL':
                # ref_obj_orientation = refscan[:, :, ::-1] * mask[:, :, ::-1]
                ref_obj_orientation = refscan[:, :, ::-1]
            if self.prep_direction == 'AP':
                # ref_obj_orientation = refscan[:, ::-1, ::-1] * mask[:, ::-1, ::-1]
                ref_obj_orientation = refscan[:, ::-1, ::-1]
            if self.prep_direction == 'FH':
                # Dit is een gok....
                # ref_obj_orientation = refscan[:, ::-1, ] * mask[:, ::-1, ]
                ref_obj_orientation = refscan[:, ::-1, ]
        elif self.target_sub_dir == 'sa':
            if self.orientation_dict['Orientation (0, 1, 2, 3)'] is not None:
                if self.orientation_dict['Orientation (0, 1, 2, 3)'] == 0:
                    # ref_obj_orientation = refscan[:, ::-1, ::-1] * mask[:, ::-1, ::-1]
                    ref_obj_orientation = refscan[:, ::-1, ::-1]
                elif self.orientation_dict['Orientation (0, 1, 2, 3)'] == 1:
                    # ref_obj_orientation = refscan[:, :, ::-1] * mask[:, :, ::-1]
                    ref_obj_orientation = refscan[:, :, ::-1]
                elif self.orientation_dict['Orientation (0, 1, 2, 3)'] == 2:
                    # ref_obj_orientation = refscan[:, ::-1, :] * mask[:, ::-1, :]
                    ref_obj_orientation = refscan[:, ::-1, :]
                elif self.orientation_dict['Orientation (0, 1, 2, 3)'] == 3:
                    # ref_obj_orientation = refscan * mask
                    ref_obj_orientation = refscan

        unfolded_img = self.sense_obj.unfold_mp(folded_img, ref_obj_orientation, self.sense_factor)
        return unfolded_img

    def get_selected_file(self, sel_index=None, sel_file_name=None):
        if (sel_index is None) and (sel_file_name is None):
            raise ValueError('Please supply atleast one None valued parameter')

        sel_file = ''
        if sel_index is not None:
            sel_index, sel_file = self.cine_file_str[sel_index]
        if sel_file_name is not None:
            filtered_list = [x for x in self.cine_file_str if sel_file_name in x[1]]
            if len(filtered_list) == 1:
                sel_index, sel_file = filtered_list[0]
            else:
                print('Somehow.. we got multiple results: ')
                for i_file in filtered_list:
                    print(i_file)

        return sel_index, sel_file

    def load_folded_file_parameters(self, cpx_obj):
        acq_param = cpx_obj.get_par_file()
        acq_img = cpx_obj.get_cpx_img()
        acq_img = acq_img[-self.sel_number_coils:]

        cine_x, cine_y = acq_img.shape[-2:]
        target_size = (cine_x * self.sense_factor, cine_y)

        if (cine_x / cine_y == 1.0) and self.target_sub_dir == 'transverse':
            print('We have found a sneeky hidden radial acquisition..')
            self.target_sub_dir = 'transverse_radial'

        return acq_param, acq_img, target_size

    def load_scan_obj(self, scan_file):
        cpx_obj_acq = read_cpx.ReadCpx(scan_file)

        target_sub_dir = self.get_target_folder(scan_file)
        if target_sub_dir is None:
            warnings.warn("Unkown image acquisition")

        if cpx_obj_acq.header is None:
            print('** Unable to read header of ', scan_file)
            target_sub_dir = None

        return cpx_obj_acq, target_sub_dir

    @staticmethod
    def get_manual_orientations():
        # These files below are created by myself....
        # The purpose is to have a proper orientation of the final image. Only a post-proc step
        data_dir = '/media/bugger/MyBook/data/7T_data/cardiac_unfolding_orientations'
        transverse_orientation_file = os.path.join(data_dir, "transverse_orientations.ods")
        transverse_orientation_table = read_ods(transverse_orientation_file, 'Sheet1')

        p2ch_orientation_file = os.path.join(data_dir, "p2ch_orientations.ods")
        p2ch_orientation_table = read_ods(p2ch_orientation_file, 'Sheet1')

        ch4_orientation_file = os.path.join(data_dir, "4ch_orientations.ods")
        ch4_orientation_table = read_ods(ch4_orientation_file, 'Sheet1')

        sa_orientation_file = os.path.join(data_dir, "sa_orientations.ods")
        sa_orientation_table = read_ods(sa_orientation_file, 'Sheet1')

        orientation_table = pd.concat([transverse_orientation_table,
                                       p2ch_orientation_table,
                                       ch4_orientation_table,
                                       sa_orientation_table], axis=0)
        orientation_table = orientation_table.reset_index(drop=True)
        orientation_table = orientation_table.replace({np.nan: None})
        return orientation_table

    @staticmethod
    def get_target_folder(file):
        # Decides WHERE we are going to store the image
        if (('trans' in file) or ('cine' in file)) and not 'radial' in file:
            target_dir = 'transverse'
        elif (('trans' in file) or ('cine' in file)) and 'radial' in file:
            target_dir = 'transverse_radial'
        elif '2ch' in file and not 'radial' in file:
            target_dir = 'p2ch'
        elif '2ch' in file and 'radial' in file:
            target_dir = 'p2ch_radial'
        elif '4ch' in file and not 'radial' in file:
            target_dir = '4ch'
        elif '4ch' in file and 'radial' in file:
            target_dir = '4ch_radial'
        elif 'sa' in file and not 'radial' in file:
            target_dir = 'sa'
        elif 'sa' in file and 'radial' in file:
            target_dir = 'sa_radial'
        elif 'survey' in file and 'radial' not in file:
            target_dir = 'survey'
        elif 'survey' in file and 'radial' in file:
            target_dir = 'survey_radial'
        elif 'shimseries' in file:
            target_dir = 'shimseries'
        elif 'sense' in file:
            target_dir = 'refscan'
        else:
            print('Unknown target type.. ', file)
            target_dir = None

        return target_dir

    @staticmethod
    def get_files_vnumber(path, v_number, extension_filter='cpx'):
        mri_files = []
        for d, sd, f in os.walk(path):
            prefix_scan_number = v_number[:2]
            regex_vnumber = re.findall(f'{prefix_scan_number}_[0-9]*', d)
            if regex_vnumber:
                if regex_vnumber[0] == v_number:
                    mri_files = [os.path.join(d, x) for x in f if x.endswith(extension_filter)]
                    mri_files = sorted(mri_files)

        return mri_files

