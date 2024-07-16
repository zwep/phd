import helper.array_transf
from objective.postproc_base import PostProcBase
import cv2
import nibabel
import helper.array_transf as harray
import skimage.transform as sktransform
from skimage.util import img_as_ubyte
import helper.misc as hmisc
import helper.plot_class as hplotc
import torch
import numpy as np
import h5py
import os
import tensorflow as tf
from data_generator.Generic import transform_array
from loguru import logger
from objective_configuration.inhomog_removal import LOG_DIR, INHOMONET_PATH, INHOMONET_WEIGHTS

try:
    file_base_name = hmisc.get_base_name(__file__)
    logger.add(os.path.join(LOG_DIR, f"{file_base_name}.log"))
except NameError:
    print('No file name known. Not reporting to logger.')


class PostProcInhomogRemoval(PostProcBase):
    def __init__(self, executor_module, image_dir, mask_dir, dest_dir=None, target_dir=None, config_dict=None, config_path=None, config_name='config_run.json', **kwargs):
        super().__init__(executor_module=executor_module, config_dict=config_dict, config_path=config_path, config_name=config_name)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dest_dir = dest_dir
        self.storage_extension = kwargs.get('storage_extension', 'npy')
        self.mask_suffix = kwargs.get('mask_suffix', '')
        self.mask_ext = kwargs.get('mask_ext', None)
        self.experimental_postproc_both = kwargs.get('exp_both', 0)
        self.target_dir = target_dir
        if self.dest_dir is None:
            self.dest_dir = os.path.join(self.config_dict['dir']['doutput'], 'prediction')
        # In case it doesnt exist yet... make it
        if not os.path.isdir(self.dest_dir):
            os.makedirs(self.dest_dir)

        self.file_list = sorted(os.listdir(image_dir))
        self.sum_of_absolute_img = None
        self.loaded_image = None
        self.loaded_mask = None
        self.loaded_target = None
        self.n_slices = None

        self.patch_shape = kwargs.get('patch_shape', None)
        self.patch_shape_equalizer = kwargs.get('patch_shape_equalizer', None)
        self.stride = kwargs.get('stride', None)
        if self.patch_shape is None:
            self.patch_shape = (256, 256)
        if self.stride is None:
            self.stride = self.patch_shape[0] // 2

        if self.config_dict is not None:
            self.transform_type = self.config_dict['data']['transform_type']
            self.transform_type_target = self.config_dict['data']['transform_type_target']
            self.target_type = self.config_dict['data']['target_type']
            self.model_obj = self.set_model_object()

    def set_model_object(self):
        model_choice = self.config_dict['model']['model_choice']
        model_obj = None
        if model_choice == 'regular':
            model_obj = self.modelrun_obj.model_obj
        elif model_choice == 'gan':
            model_obj = self.modelrun_obj.generator
        elif model_choice == 'cyclegan':
            model_obj = self.modelrun_obj.netG_A2B
        return model_obj

    def run_tensorflow(self):
        """
        here we are going to run the tensorflow model Inhomonet.
        We already have pretrained weights for this one..
        :return:
        """
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            # This contains the corrected, biasfield and multi-coil corrected
            file_result = self.run_loaded_file_patched_tensorflow()
            corrected_result = np.array([x['corrected'] for x in file_result])
            print('Shape of corrected result ', corrected_result.shape)
            self.save_array(corrected_result, "pred/" + i_file, self.storage_extension)
            biasfield = np.array([x['biasfield'] for x in file_result])
            self.save_array(biasfield, 'biasfield/' + i_file, self.storage_extension)
            self.save_array(self.sum_of_absolute_img, 'input/' + i_file, self.storage_extension)
            self.save_array(self.loaded_mask, 'mask/' + i_file, self.storage_extension)
            if self.loaded_target is not None:
                self.save_array(self.loaded_target, 'target/' + i_file, self.storage_extension)

    def run(self):
        # Do a complete run...fixed to to a patch-based evaluation
        # This is done for the datasets where we dont have a target image
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            # This contains the corrected, biasfield and multi-coil corrected
            file_result = self.run_loaded_file_patched()
            corrected_result = np.array([x['corrected'] for x in file_result])
            print('Shape of corrected result ', corrected_result.shape)
            self.save_array(corrected_result, "pred/" + i_file, self.storage_extension)
            biasfield = np.array([x['biasfield'] for x in file_result])
            self.save_array(biasfield, 'biasfield/' + i_file, self.storage_extension)
            self.save_array(self.sum_of_absolute_img, 'input/' + i_file, self.storage_extension)
            self.save_array(self.loaded_mask, 'mask/' + i_file, self.storage_extension)
            if self.loaded_target is not None:
                self.save_array(self.loaded_target, 'target/' + i_file, self.storage_extension)

    def run_iterative_recon(self):
        # Do a complete run...fixed to to a patch-based evaluation
        # Here we perform an iterative reconstruction
        total_iter_relative = []
        total_iter_abs = []
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            iter_rmse_relative, iter_rmse_abs = self.iterative_reconstruction(self.n_slices//2)
            total_iter_relative.append(iter_rmse_relative)
            total_iter_abs.append(iter_rmse_abs)
        dest_file_abs = os.path.join(self.dest_dir, 'abs_iterative.csv')
        dest_file_rel = os.path.join(self.dest_dir, 'rel_iterative.csv')
        import csv

        with open(dest_file_abs, "w") as f:
            wr = csv.writer(f)
            wr.writerows(total_iter_abs)

        with open(dest_file_rel, "w") as f:
            wr = csv.writer(f)
            wr.writerows(total_iter_relative)

    def save_array(self, x_img, file_name, extension):
        assert self.dest_dir is not None
        print('File name before base_name ', file_name) 
        dir_name = os.path.dirname(file_name)
        file_name = hmisc.get_base_name(file_name)
        print('After base_name ', file_name)
        print('Current dest_dir ', self.dest_dir)
        dest_file = os.path.join(self.dest_dir, dir_name, file_name)
        # x_img = harray.treshold_percentile(x_img, q=98)
        x_img = harray.scale_minmax(x_img)
        x_img = img_as_ubyte(x_img)
        if 'npy' in extension:
            np.save(dest_file + '.npy', x_img)
        elif 'nii' in extension:
            x_img = x_img.T[::-1, ::-1]
            nibabel_result = nibabel.Nifti1Image(x_img, np.eye(4))
            print('Target path... ', dest_file)
            print('File name...', file_name)
            nibabel.save(nibabel_result, dest_file + '.nii.gz')
        elif 'h5' in extension:
            with h5py.File(dest_file + '.h5', 'w') as f:
                f.create_dataset('data', data=x_img)
        else:
            print("Unknown data extension: ", extension)
            print("Please use npy or h5")

    def load_file(self, file_index):
        # Twijfel een beetje aan de file counter...
        sel_file = self.file_list[file_index]
        sel_img = os.path.join(self.image_dir, sel_file)
        # Is this always the case? Dont I need some appendix sometime for the mask files?
        sel_file_name = hmisc.get_base_name(sel_file)
        sel_file_ext = hmisc.get_ext(sel_file)
        if self.mask_ext is not None:
            sel_file_ext = self.mask_ext
        if self.mask_dir is not None:
            sel_mask = os.path.join(self.mask_dir, sel_file_name + self.mask_suffix + sel_file_ext)
        else:
            sel_mask = ''

        if self.target_dir is not None:
            sel_target_file = os.path.join(self.target_dir, sel_file_name + sel_file_ext)
            self.loaded_target = hmisc.load_array(sel_target_file)
            if sel_target_file.endswith('nii.gz'):
                self.loaded_target = self.loaded_target.T[:, ::-1, ::-1]

        # Stuff...
        self.loaded_image = hmisc.load_array(sel_img)
        if sel_mask:
            self.loaded_mask = hmisc.load_array(sel_mask)
        else:
            self.loaded_mask = np.ones(self.loaded_image.shape)
        if sel_img.endswith('nii.gz'):
            self.loaded_image = self.loaded_image.T[:, ::-1, ::-1]
        if sel_mask.endswith('nii.gz'):
            self.loaded_mask = self.loaded_mask.T[:, ::-1, ::-1]

        # Make sure that we deal with NUMBERS, not with booleans.
        # HOWEVER, this now goes wrong when we have only ones..
        self.loaded_mask = harray.scale_minmax(self.loaded_mask.astype(np.uint8))
        self.loaded_mask[np.isnan(self.loaded_mask)] = 1
        # Data from patient data is of shape (loc, nx, ny)
        # Data from volunteer is shape (chan, nx, ny)
        # Both format NPY

        # Data from test split is shape (loc, nx, ny)
        # Or.. (cpx, chan, loc, nx, ny)
        # Either NPY or nii

        # The test split-numpy images are 5 dimensional
        # (re/im, coil, loc, nx, ny)
        if self.loaded_image.ndim == 5:
            self.loaded_image = self.loaded_image[0] + 1j * self.loaded_image[1]

        n_chan_or_n_slice = self.loaded_image.shape[0]
        if n_chan_or_n_slice == 8:
            self.n_chan = 8
            if self.loaded_image.ndim == 4:
                # This changes the dimensions to,...
                # (loc, coil, nx, ny)
                self.loaded_image = np.moveaxis(self.loaded_image, 0, 1)
                self.n_slices = self.loaded_image.shape[0]
                # On axis 1 we have the number of coils...
                self.sum_of_absolute_img = np.abs(self.loaded_image).mean(axis=1)
                # Make the img size (n_loc, n_chan, nx ny)
                self.loaded_image = self.loaded_image
            else:
                self.n_slices = 1
                # Make the img size (n_loc=1, nx ny) -- note that [None] at the end
                self.sum_of_absolute_img = np.abs(self.loaded_image).mean(axis=0)[None]
                # Make the img size (n_loc=1, n_chan, nx ny)
                self.loaded_image = self.loaded_image[None]
                # Make the img size (n_loc=1, nx ny)
                self.loaded_mask = self.loaded_mask[None]
            # These locations are always 1....
            # Check if we have a complex img....
            is_complex = np.iscomplex(self.loaded_image.ravel()[0])
            self.loaded_image = harray.scale_minmax(self.loaded_image, is_complex=is_complex)
        else:
            self.n_chan = 1
            self.n_slices = n_chan_or_n_slice
            self.sum_of_absolute_img = np.abs(self.loaded_image)
            # Make the img size (n_loc, n_chan=1, nx ny)
            self.loaded_image = self.loaded_image[:, None]
            # Scale it per location to min/max
            self.loaded_image = harray.scale_minmax(self.loaded_image, axis=(-2, -1))
            # Keep the img size (n_loc, nx ny)
            self.loaded_mask = self.loaded_mask

        self.sum_of_absolute_img = harray.scale_minmax(self.sum_of_absolute_img)

        print("Result of loaded file ")
        print("Size of image array ", self.loaded_image.shape)
        print("\tNumber of slices ", self.n_slices)
        print("\tNumber of channels ", self.n_chan)
        print("Size of mask array ", self.loaded_mask.shape)
        print("Size of sum of abs ", self.sum_of_absolute_img.shape)

    def run_loaded_file_full_image(self):
        file_result = []
        # Wil ik altijd alle slices doen...?
        for i_slice in range(self.n_slices):
            slice_result = self.run_slice_full_image(slice_index=i_slice)
            postproc_slice = self.postproc_loaded(slice_result, slice_index=i_slice)
            file_result.append(postproc_slice)

        return file_result

    def run_loaded_file_patched_tensorflow(self):
        file_result = []
        for i_slice in range(self.n_slices):
            print(f'Running slice {i_slice} / {self.n_slices}', end='\r')
            slice_result = self.run_slice_patched_tensorfow(slice_index=i_slice)
            postproc_slice = self.postproc_loaded(slice_result, slice_index=i_slice)
            file_result.append(postproc_slice)
        print()
        return file_result

    def run_loaded_file_patched(self):
        file_result = []
        for i_slice in range(self.n_slices):
            print(f'Running slice {i_slice} / {self.n_slices}', end='\r')
            slice_result = self.run_slice_patched(slice_index=i_slice)
            postproc_slice = self.postproc_loaded(slice_result, slice_index=i_slice)
            file_result.append(postproc_slice)
        print()
        return file_result

    def run_slice_patched_tensorfow(self, slice_index):
        sel_slice = self.loaded_image[slice_index]
        sel_mask = self.loaded_mask[slice_index]
        # Transform to right type..
        # Squeeze this.. because Tensorflow model doesnt want (1, ny, nx) as input..
        sel_slice = transform_array(sel_slice, transform_type=self.transform_type)
        #
        sel_slice_result = self.run_patches_tensorflow(sel_slice, sel_mask)
        return sel_slice_result

    def run_slice_patched(self, slice_index):
        sel_slice = self.loaded_image[slice_index]
        sel_mask = self.loaded_mask[slice_index]
        # Transform to right type..
        sel_slice = transform_array(sel_slice, transform_type=self.transform_type)
        # What if we put the 1 value at the mask locations
        # Those should not be corrected.. maybe edges are better preserved then
        # print('\n\nmean before ', np.mean(sel_slice))
        # print("Shape of sel slice ", sel_slice.shape)
        # for ii in range(sel_slice.shape[0]):
        #     sel_slice[ii][sel_mask == 0] = 0
        # print('mean after ', np.mean(sel_slice))
        sel_slice_result = self.run_patches(sel_slice, sel_mask)
        return sel_slice_result

    def run_slice_full_image(self, slice_index):
        sel_slice = self.loaded_image[slice_index]
        sel_mask = self.loaded_mask[slice_index]
        # Transform to right type..
        sel_slice = transform_array(sel_slice, transform_type=self.transform_type)

        sel_slice_result = self.run_full_image(sel_slice[None], sel_mask)

        return sel_slice_result

    def run_full_image(self, x_img, x_mask):
        input_tensor = torch.from_numpy(x_img).float()
        input_tensor.to(self.modelrun_obj.device)
        with torch.no_grad():
            result_model = self.model_obj(input_tensor)

        if self.modelrun_obj.device == "cpu":
            result_model = result_model.numpy()
        else:
            # Then we assume CUDA..
            result_model = result_model.cpu().numpy()

        result_model_np = result_model[0][0]
        if result_model_np.shape != x_mask.shape:
            result_model_np = sktransform.resize(result_model_np, self.sum_of_absolute_img.shape)

        result_model_np = result_model_np * x_mask

        return result_model_np

    def init_tensorflow(self):
        """
        Restore the graph..
        :return:
        """

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        # restore weights
        tf.compat.v1.disable_eager_execution()
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(INHOMONET_WEIGHTS, 'model_final.ckpt.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(INHOMONET_WEIGHTS))

        graph = tf.compat.v1.get_default_graph()
        Y_real = graph.get_tensor_by_name('Y:0')
        X_real = graph.get_tensor_by_name('X:0')
        Y_fake = graph.get_tensor_by_name('G_xy/conv2d_78/Relu:0')
        self.tf_session = sess
        self.Y_real = Y_real
        self.X_real = X_real
        self.Y_fake = Y_fake

    def run_patches_tensorflow(self, x_img, x_mask):
        # First.. correct for the size..
        orig_shape = x_img.shape
        pad_x, pad_y = np.array(orig_shape[-2:]) % self.patch_shape[0]
        # Now the image can have an integer number of patches...
        x_img = np.pad(x_img, ((0, 0), (0, pad_x), (0, pad_y)))
        x_mask = np.pad(x_mask, ((0, pad_x), (0, pad_y)))
        img_patches = [harray.get_patches(x, patch_shape=self.patch_shape, stride=self.stride) for x in x_img]
        img_patches = np.stack(img_patches, axis=1)
        mask_patches = harray.get_patches(x_mask, patch_shape=self.patch_shape, stride=self.stride)
        n_patches = len(mask_patches)

        result_patches = []
        # This one is needed when we predict both biasfield AND homogeneous img
        second_result_patches = []
        for i_patch in range(n_patches):
            i_mask_patch = mask_patches[i_patch]
            i_array_patch = img_patches[i_patch]
            i_array_patch = harray.scale_minmax(i_array_patch) * i_mask_patch
            i_array_patch = i_array_patch[:, :, :, None]

            result_model = self.tf_session.run([self.Y_fake], feed_dict={self.X_real: i_array_patch})
            result_model_np = result_model[0][0, :, :, 0]
            result_patches.append(result_model_np)
            if self.target_type == 'both':
                # Ik gok dat het `tweede` kanaal dat daar bias field in zit
                second_result_patches.append(result_model[0][0, :, :, 1])

        result_stiched = harray.get_stiched(result_patches, x_img.shape[-2:],
                                            patch_shape=self.patch_shape, stride=self.stride)
        result_stiched = helper.array_transf.correct_inf_nan(result_stiched)
        # Revert back to original size
        result_stiched = result_stiched[:orig_shape[-2], :orig_shape[-1]]
        if self.target_type == 'both':
            second_result_stiched = harray.get_stiched(second_result_patches, x_img.shape[-2:],
                                                patch_shape=self.patch_shape, stride=self.stride)
            second_result_stiched = helper.array_transf.correct_inf_nan(second_result_stiched)
            # Revert back to original size
            second_result_stiched = second_result_stiched[:orig_shape[0], :orig_shape[1]]
            return result_stiched, second_result_stiched
        else:
            return result_stiched

    def run_patches(self, x_img, x_mask):
        # Er is wat voor te zeggen deze alleen te laten draaien op input Ximg en Xmask
        # Maar ergens voelt t wat kort door de bocht.. en krijg ik veel bloat methods
        # ?? Geen idee wat ik met bovenstaande bedoelde.
        img_patches = [harray.get_patches(x, patch_shape=self.patch_shape, stride=self.stride) for x in x_img]
        img_patches = np.stack(img_patches, axis=1)
        mask_patches = harray.get_patches(x_mask, patch_shape=self.patch_shape, stride=self.stride)
        n_patches = len(mask_patches)

        result_patches = []
        # This one is needed when we predict both biasfield AND homogeneous img
        second_result_patches = []
        for i_patch in range(n_patches):
            i_mask_patch = mask_patches[i_patch]
            i_array_patch = img_patches[i_patch]
            i_array_patch = harray.scale_minmax(i_array_patch) * i_mask_patch

            input_tensor = torch.as_tensor(i_array_patch[np.newaxis]).float()
            input_tensor = input_tensor.to(self.modelrun_obj.device)
            i_mask_patch_tensor = torch.as_tensor(i_mask_patch).float()
            i_mask_patch_tensor = i_mask_patch_tensor.to(self.modelrun_obj.device)

            with torch.no_grad():
                result_model = self.model_obj(input_tensor) * i_mask_patch_tensor

            if self.modelrun_obj.device == "cpu":
                result_model = result_model.numpy()
            else:
                result_model = result_model.cpu().numpy()
            # This probably has more channels..?
            result_model_np = result_model[0][0]
            result_patches.append(result_model_np)
            if self.target_type == 'both':
                # Ik gok dat het `tweede` kanaal dat daar bias field in zit
                second_result_patches.append(result_model[0][1])

        result_stiched = harray.get_stiched(result_patches, x_img.shape[-2:],
                                            patch_shape=self.patch_shape, stride=self.stride)
        result_stiched = helper.array_transf.correct_inf_nan(result_stiched)
        if self.target_type == 'both':
            second_result_stiched = harray.get_stiched(second_result_patches, x_img.shape[-2:],
                                                patch_shape=self.patch_shape, stride=self.stride)
            second_result_stiched = helper.array_transf.correct_inf_nan(second_result_stiched)
            return result_stiched, second_result_stiched
        else:
            return result_stiched

    def postproc_loaded(self, result, slice_index=0):
        sel_slice = self.loaded_image[slice_index]
        # Transform to right type..
        transformed_slice = transform_array(sel_slice, transform_type=self.transform_type)
        if self.target_type == 'biasfield':
            bias_field = result * self.loaded_mask[slice_index]
            bias_field = harray.smooth_image(np.squeeze(bias_field), n_kernel=bias_field.shape[-1] // 32)
            corrected_image = self.sum_of_absolute_img[slice_index] / (1+bias_field)
            # corrected_image = np.log(self.sum_of_absolute_img[slice_index]) - np.log((0.001 + bias_field))
            #corrected_image = np.exp(corrected_image)
            corrected_image = helper.array_transf.correct_inf_nan(corrected_image) * self.loaded_mask[slice_index]
        elif self.target_type == 'both':
            # Now we have two predictinos.. both the homogeneous AND the bias field.
            # We can do perform four predictions...
            pred_homog_image, pred_bias_field = result
            if self.experimental_postproc_both == 0:
                # Use only the homogeneous result
                calc_bias_field = self.sum_of_absolute_img[slice_index] / pred_homog_image
                calc_bias_field = helper.array_transf.correct_inf_nan(calc_bias_field) * self.loaded_mask[slice_index]
                calc_bias_field = harray.treshold_percentile_both(calc_bias_field, q=99)
                calc_bias_field = harray.scale_minmax(calc_bias_field)
                bias_field = harray.smooth_image(np.squeeze(calc_bias_field), n_kernel=calc_bias_field.shape[-1] // 32)
                corrected_image = self.sum_of_absolute_img[slice_index] / bias_field * self.loaded_mask[slice_index]
                corrected_image = helper.array_transf.correct_inf_nan(corrected_image) * self.loaded_mask[slice_index]
            elif self.experimental_postproc_both == 1:
                # Use only the bias field result
                pred_bias_field = pred_bias_field * self.loaded_mask[slice_index]
                bias_field = harray.smooth_image(np.squeeze(pred_bias_field), n_kernel=pred_bias_field.shape[-1] // 32)
                corrected_image = self.sum_of_absolute_img[slice_index] / (1 + bias_field)
                # corrected_image = np.log(self.sum_of_absolute_img[slice_index]) - np.log((0.001 + bias_field))
                # corrected_image = np.exp(corrected_image)
                corrected_image = helper.array_transf.correct_inf_nan(corrected_image) * self.loaded_mask[slice_index]
            elif self.experimental_postproc_both == 2:
                # Calculate a bias field from the predicted homogeneous
                # Average that with the predicted bias field
                # Continue
                calc_bias_field = self.sum_of_absolute_img[slice_index] / pred_homog_image
                calc_bias_field = helper.array_transf.correct_inf_nan(calc_bias_field) * self.loaded_mask[slice_index]
                calc_bias_field = harray.treshold_percentile_both(calc_bias_field, q=99)
                bias_field = (calc_bias_field + pred_bias_field) / 2
                # bias_field = harray.scale_minmax(bias_field)
                bias_field = harray.smooth_image(np.squeeze(bias_field), n_kernel=bias_field.shape[-1] // 32)
                corrected_image = self.sum_of_absolute_img[slice_index] / (1 + bias_field)
                corrected_image = helper.array_transf.correct_inf_nan(corrected_image) * self.loaded_mask[slice_index]
            elif self.experimental_postproc_both == 3:
                # Calculate a corrected imagefrom the predicted bias field
                # Average that with the predicted homogeneous image
                # Continue
                pred_bias_field = pred_bias_field * self.loaded_mask[slice_index]
                # bias_field = harray.smooth_image(np.squeeze(pred_bias_field), n_kernel=pred_bias_field.shape[-1] // 32)
                calc_image = self.sum_of_absolute_img[slice_index] / (1 + pred_bias_field)
                calc_image_avg = (calc_image + pred_homog_image) / 2
                # Now get the calculated bias field from both.. and THEN do this..
                calc_bias_field = self.sum_of_absolute_img[slice_index] / calc_image_avg
                calc_bias_field = helper.array_transf.correct_inf_nan(calc_bias_field) * self.loaded_mask[slice_index]
                calc_bias_field = harray.treshold_percentile_both(calc_bias_field, q=99)
                bias_field = harray.smooth_image(np.squeeze(calc_bias_field), n_kernel=pred_bias_field.shape[-1] // 32)
                corrected_image = self.sum_of_absolute_img[slice_index] / bias_field * self.loaded_mask[slice_index]
                corrected_image = helper.array_transf.correct_inf_nan(corrected_image) * self.loaded_mask[slice_index]
            else:
                bias_field = None
                corrected_image = None
        else:
            bias_field = self.sum_of_absolute_img[slice_index] / result
            bias_field = helper.array_transf.correct_inf_nan(bias_field) * self.loaded_mask[slice_index]
            bias_field = harray.treshold_percentile_both(bias_field, q=98)
            bias_field = harray.scale_minmax(bias_field)
            bias_field = harray.smooth_image(np.squeeze(bias_field), n_kernel=bias_field.shape[-1] // 32)
            corrected_image = self.sum_of_absolute_img[slice_index] / (bias_field * self.loaded_mask[slice_index])
            corrected_image = helper.array_transf.correct_inf_nan(corrected_image) * self.loaded_mask[slice_index]
            corrected_image = harray.treshold_percentile_both(corrected_image, q=98)
            # TODO: The idea here is to extract the fat layer/outside layer and
            # treshold based on that.
            shrunk_mask = harray.shrink_image(self.loaded_mask[slice_index], n_pixel=10, order=0)


        # The corrected multi is needed to facilitate the iterative correction procedure
        corrected_multi_coil_img = transformed_slice / (1+bias_field)
        corrected_multi_coil_img = helper.array_transf.correct_inf_nan(corrected_multi_coil_img) * self.loaded_mask[slice_index]
        return {'corrected': corrected_image, 'biasfield': bias_field, 'corrected_multi': corrected_multi_coil_img}

    def iterative_reconstruction(self, slice_index, n_iter=10, store_intermediate_recon=False, prefix_name='', file_name=''):
        # This has simply not been implemented yet. Not sure if I want to continue that...
        assert self.target_type != 'both'
        iter_rmse_relative = []
        iter_rmse_abs = []
        slice_result = self.run_slice_patched(slice_index=slice_index)
        postproc_slice = self.postproc_loaded(slice_result, slice_index=slice_index)
        if self.n_chan == 8:
            temp_corrected = postproc_slice['corrected_multi']
        else:
            temp_corrected = postproc_slice['corrected']
        if temp_corrected.ndim == 2:
            temp_corrected = temp_corrected[None]

        initial_biasf = postproc_slice['biasfield']
        prev_biasf = postproc_slice['biasfield']
        #
        n_slice = temp_corrected.shape[0]
        n_midy = temp_corrected.shape[-1] // 2
        n_midx = temp_corrected.shape[-2] // 2
        # And now crop..
        crop_coords = (n_midx, n_midx, n_midy, n_midy)
        zoom_size = 125
        #
        if store_intermediate_recon:
            iter_nr = 0
            n_slice = temp_corrected.shape[0]
            temp_cropped = harray.apply_crop(temp_corrected[n_slice // 2], crop_coords=crop_coords, marge=zoom_size)
            fig_obj = hplotc.ListPlot([temp_corrected[n_slice // 2], temp_cropped, prev_biasf], title=f'iter_{iter_nr}_{file_name}',
                                      cbar=True)
            nifti_obj = nibabel.Nifti1Image(temp_corrected[n_slice // 2][None].T[::-1, ::-1], np.eye(4))
            nibabel.save(nifti_obj,
                         f'/home/sharreve/local_scratch/temp_data_storage/iter_recon/{prefix_name}_iter_{iter_nr}.nii.gz')
            fig_obj.figure.savefig(
                f'/home/sharreve/local_scratch/temp_data_storage/iter_recon/{prefix_name}_iter_{iter_nr}.png')
        for iter_nr in range(1, n_iter):
            print(iter_nr, end='\r')
            if self.n_chan == 8:
                temp_corrected = postproc_slice['corrected_multi']
            else:
                temp_corrected = postproc_slice['corrected']

            # Scale the corrected image...
            nx, ny = temp_corrected.shape[-2:]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nx // 32, ny // 32))
            temp_corrected = harray.scale_minmax(temp_corrected)
            temp_corrected = img_as_ubyte(temp_corrected)
            print("SHAPE", temp_corrected.shape)
            if temp_corrected.ndim == 2:
                temp_corrected = clahe.apply(temp_corrected)
                temp_corrected = harray.scale_minmax(temp_corrected)
                temp_corrected = temp_corrected[None]
            else:
                temp_corrected = np.array([clahe.apply(x) for x in temp_corrected])
                temp_corrected = harray.scale_minmax(temp_corrected)
            # temp_corrected = harray.treshold_percentile(temp_corrected, q=98)
            if store_intermediate_recon:
                temp_cropped = harray.apply_crop(temp_corrected[n_slice//2], crop_coords=crop_coords, marge=zoom_size)
                fig_obj = hplotc.ListPlot([temp_corrected[n_slice//2], temp_cropped, prev_biasf], title=f'iter_{iter_nr}_{file_name}', cbar=True)
                nifti_obj = nibabel.Nifti1Image(temp_corrected[n_slice // 2][None].T[::-1, ::-1], np.eye(4))
                nibabel.save(nifti_obj, f'/home/sharreve/local_scratch/temp_data_storage/iter_recon/{prefix_name}_iter_{iter_nr}.nii.gz')
                fig_obj.figure.savefig(f'/home/sharreve/local_scratch/temp_data_storage/iter_recon/{prefix_name}_iter_{iter_nr}.png')

            temp_result = self.run_patches(x_img=temp_corrected, x_mask=self.loaded_mask[slice_index])
            postproc_slice = self.postproc_loaded(temp_result, slice_index=slice_index)
            temp_value = np.mean((prev_biasf - postproc_slice['biasfield']) ** 2)
            temp_abs_value = np.mean((initial_biasf - postproc_slice['biasfield']) ** 2)
            prev_biasf = postproc_slice['biasfield']
            iter_rmse_relative.append(temp_value)
            iter_rmse_abs.append(temp_abs_value)

        return iter_rmse_relative, iter_rmse_abs
