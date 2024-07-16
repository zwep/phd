import json
import matplotlib.pyplot as plt
import skimage.metrics
import multiprocessing as mp
# SOURCE: https://pythonspeed.com/articles/python-multiprocessing/
import helper.misc as hmisc
import helper.array_transf as harray
import scipy.stats
import small_project.homogeneity_measure.metric_implementations as homog_metric
import os
import numpy as np
import skimage.feature
import helper.plot_class as hplotc
from loguru import logger
from objective_configuration.inhomog_removal import LOG_DIR
import sys


try:
    # Check if we have a __file__ name
    file_base_name = hmisc.get_base_name(__file__)
    logger.add(os.path.join(LOG_DIR, f"{file_base_name}.log"))
except NameError:
    print('No file name known. Not reporting to logger.')


class CalculateMetrics:
    def __init__(self, dimage, dpred, **kwargs):
        logger.info('Initializing CalculateMetrics')
        # So... somehow certain masks are not precssed properly
        self.debug = kwargs.get('debug', False)  # Reports addition image metrics if True
        self.dimage = dimage
        self.dpred = dpred
        self.dmask = kwargs.get('dmask', None)
        # If dtarget is None is used to check if we have a target file
        self.dtarget = kwargs.get('dtarget', None) if len(os.listdir(kwargs.get('dtarget', None))) else None
        # Optional mask files
        # Here we need an empty string as default since we are joining paths with it
        self.dmask_prostate = kwargs.get('dmask_prostate', '')
        self.dmask_fat = kwargs.get('dmask_fat', '')
        self.dmask_muscle = kwargs.get('dmask_muscle', '')
        self.file_list = sorted(os.listdir(dimage))

        # Sometimes mask files have some suffix...
        self.mask_suffix = kwargs.get('mask_suffix', '')
        self.mask_ext = kwargs.get('mask_ext', '')
        self.patch_size = kwargs.get('patch_size', None)
        self.slice_patch_size = None
        self.mid_slice = kwargs.get('mid_slice', False)
        self.shrink_pixels = kwargs.get('shrink_pixels', 0)

        # Parameters for GLCM stuff
        self.glcm_dist = kwargs.get('glcm_dist', [1, 2])
        self.max_slices = 30

        # Predefine any image we might load
        self.loaded_image = None
        self.loaded_image_slice = None
        self.loaded_target = None
        self.loaded_target_slice = None
        self.loaded_pred = None
        self.loaded_pred_slice = None
        self.loaded_mask = None
        self.loaded_mask_prostate = None
        self.loaded_mask_fat = None
        self.loaded_mask_muscle = None
        self.loaded_mask_slice = None
        self.n_slices = None
        self.n_cores = mp.cpu_count()
        self.feature_keys = kwargs.get('feature_keys', None)
        if self.feature_keys is None:
            self.feature_keys = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    def print_features_current_slice(self):
        # Now calculate current values..
        temp_pred_slice, temp_mask_slice = harray.get_crop(self.loaded_pred_slice, self.loaded_mask_slice)
        temp_image_slice, _ = harray.get_crop(self.loaded_image_slice, self.loaded_mask_slice)
        rel_temp_dict, pred_temp_dict, input_temp_dict = self.get_glcm_slice(temp_pred_slice,
                                                                             temp_image_slice,
                                                                             patch_size=self.slice_patch_size)
        # Get relative COV features
        temp_inp = temp_image_slice[temp_mask_slice != 0]
        temp_pred = temp_pred_slice[temp_mask_slice != 0]
        coeff_var_input_value = np.std(temp_inp) / np.mean(temp_inp)
        coeff_var_pred_value = np.std(temp_pred) / np.mean(temp_pred)
        coeff_var_rel_value = (coeff_var_pred_value - coeff_var_input_value) / coeff_var_input_value
        logger.info("Relative GLCM dict ")
        hmisc.print_dict(rel_temp_dict)
        logger.info("Relative Coeficient of variation ", coeff_var_rel_value)
        return {'glcm': (rel_temp_dict, pred_temp_dict, input_temp_dict),
                'cov': (coeff_var_input_value, coeff_var_pred_value, coeff_var_rel_value)}

    def print_target_features_current_slice(self):
        if self.dtarget is not None:
            rmse_value = np.sqrt(np.mean((self.loaded_pred_slice - self.loaded_target_slice) ** 2))
            pred_hist, _ = np.histogram(self.loaded_pred_slice.ravel(), bins=256, range=(0, 1))
            target_hist, _ = np.histogram(self.loaded_target_slice.ravel(), bins=256, range=(0, 1))
            wss_value = scipy.stats.wasserstein_distance(pred_hist, target_hist)
            ssim_target_pred = skimage.metrics.structural_similarity(self.loaded_pred_slice, self.loaded_target_slice, data_range=255)
            logger.info("Wasserstein distance ", wss_value)
            logger.info("SSIM ", ssim_target_pred)
            logger.info("RMSE ", rmse_value)
            return rmse_value, wss_value, ssim_target_pred

    def save_current_slice(self, storage_name):
        # Will always be stored at a specific location
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        n = 256
        input_hist, bins_numpy = np.histogram(self.loaded_image_slice[self.loaded_mask_slice == 1].ravel(), bins=n)
        ax[0].bar(bins_numpy[:-1], input_hist, width=1, color='b', alpha=0.5, label='input')
        ax[1].imshow(self.loaded_image_slice * self.loaded_mask_slice)
        ax[1].set_title('input')
        pred_hist, _ = np.histogram(self.loaded_pred_slice[self.loaded_mask_slice == 1].ravel(), bins=n)
        ax[2].imshow(self.loaded_pred_slice * self.loaded_mask_slice)
        ax[2].set_title('pred')
        ax[0].bar(bins_numpy[:-1], pred_hist, width=1, color='r', alpha=0.5, label='pred')
        if self.dtarget is not None:
            target_hist, _ = np.histogram(self.loaded_target_slice[self.loaded_mask_slice == 1].ravel(), bins=n)
            ax[0].bar(bins_numpy[:-1], target_hist, width=1, color='g', alpha=0.5, label='target')
            ax[3].imshow(self.loaded_target_slice * self.loaded_mask_slice)
            ax[3].set_title('target')
        ax[0].legend()
        fig.suptitle(storage_name)
        fig.savefig(os.path.join(os.path.dirname(self.dpred), f'{storage_name}.png'))

    def load_file(self, file_index):
        # This is based on dimage
        sel_file = self.file_list[file_index]
        if len(self.mask_ext):
            sel_file_ext = self.mask_ext
        else:
            sel_file_ext = hmisc.get_ext(sel_file)
        sel_file_name = hmisc.get_base_name(sel_file)
        sel_img_file = os.path.join(self.dimage, sel_file)
        sel_mask_file = os.path.join(self.dmask, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_target_file = ''
        if self.dtarget is not None:
            sel_target_file = os.path.join(self.dtarget, sel_file)
        sel_mask_prostate_file = os.path.join(self.dmask_prostate, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_mask_fat_file = os.path.join(self.dmask_fat, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_mask_muscle_file = os.path.join(self.dmask_muscle, sel_file_name + self.mask_suffix + sel_file_ext)
        sel_pred_file = os.path.join(self.dpred, sel_file)

        # Stuff...
        img_file_ext = hmisc.get_ext(sel_img_file)
        self.loaded_image = hmisc.load_array(sel_img_file)
        logger.debug(f'Image path {sel_img_file}')
        if 'nii' in img_file_ext:
            self.loaded_image = self.loaded_image.T[:, ::-1, ::-1]

        self.loaded_pred = hmisc.load_array(sel_pred_file)
        logger.debug(f'Pred path {sel_pred_file}')
        if 'nii' in hmisc.get_ext(sel_pred_file):
            self.loaded_pred = self.loaded_pred.T[:, ::-1, ::-1]

        if os.path.isfile(sel_target_file):
            self.loaded_target = hmisc.load_array(sel_target_file)
            logger.debug(f'Target path {sel_target_file}')
            if 'nii' in hmisc.get_ext(sel_target_file):
                self.loaded_target = self.loaded_target.T[:, ::-1, ::-1]
            # self.loaded_target = harray.scale_minmax(self.loaded_target, axis=(-2, -1))

        if os.path.isfile(sel_mask_file):
            logger.debug(f'Mask path {sel_mask_file}')
            self.loaded_mask = self.load_mask(sel_mask_file)

        if os.path.isfile(sel_mask_prostate_file):
            logger.debug(f'Prostate mask path {sel_mask_prostate_file}')
            logger.debug(f'Current directory {os.getcwd()}')
            self.loaded_mask_prostate = self.load_mask(sel_mask_prostate_file)

        if os.path.isfile(sel_mask_fat_file):
            logger.debug(f'Fat mask path {sel_mask_fat_file}')
            logger.debug(f'Current directory {os.getcwd()}')
            self.loaded_mask_fat = self.load_mask(sel_mask_fat_file)

        if os.path.isfile(sel_mask_muscle_file):
            logger.debug(f'Muscle mask path {sel_mask_muscle_file}')
            logger.debug(f'Current directory {os.getcwd()}')
            self.loaded_mask_muscle = self.load_mask(sel_mask_muscle_file)

        self.n_slices = self.loaded_image.shape[0]

        # Do or dont do this..?
        # self.loaded_image = harray.scale_minmax(self.loaded_image, axis=(-2, -1))
        # self.loaded_pred = harray.scale_minmax(self.loaded_pred, axis=(-2, -1))
    def print_image_info(self):
        # Print here stuff from the loaded images.. like
        logger.info(f'Image shape {self.loaded_image.shape}')
        logger.info(f'Pred shape {self.loaded_pred.shape}')
        logger.info(f'Mask shape {self.loaded_mask.shape}')
        if self.dtarget is not None:
            logger.info(f'Target shape {self.loaded_target.shape}')

    def print_slice_info(self):
        temp_list = harray.get_minmeanmediammax(self.loaded_image_slice)
        temp_list = [str(x) for x in temp_list]
        logger.info(f"Min/Mean/Median/Max image {', '.join(temp_list)}")
        temp_list = harray.get_minmeanmediammax(self.loaded_pred_slice)
        temp_list = [str(x) for x in temp_list]
        logger.info(f"Min/Mean/Median/Max pred {', '.join(temp_list)}")
        temp_list = harray.get_minmeanmediammax(self.loaded_mask_slice)
        temp_list = [str(x) for x in temp_list]
        logger.info(f"Min/Mean/Median/Max mask {', '.join(temp_list)}")
        if self.dtarget is not None:
            temp_list = harray.get_minmeanmediammax(self.loaded_target_slice)
            temp_list = [str(x) for x in temp_list]
            logger.info(f"Min/Mean/Median/Max target {', '.join(temp_list)}")

    def set_slice(self, slice_index):
        self.loaded_mask_slice = self.loaded_mask[slice_index]
        # Some CLAHE stuff
        # Now we can do several histogram equilization things...
        import cv2
        from skimage.util import img_as_ubyte
        # nx, ny = self.loaded_mask_slice.shape
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nx // 32, ny // 32))
        # Load array...
        if self.loaded_mask_fat is not None:
            self.loaded_mask_slice = self.loaded_mask_slice * self.loaded_mask_fat[slice_index]
        self.loaded_image_slice = self.loaded_image[slice_index]
        self.loaded_image_slice = harray.scale_minmax(self.loaded_image_slice)
        self.loaded_image_slice = img_as_ubyte(self.loaded_image_slice)
        # self.loaded_image_slice = clahe.apply(self.loaded_image_slice)

        self.loaded_pred_slice = self.loaded_pred[slice_index]
        self.loaded_pred_slice = harray.scale_minmax(self.loaded_pred_slice)
        self.loaded_pred_slice = img_as_ubyte(self.loaded_pred_slice)
        # self.loaded_pred_slice = clahe.apply(self.loaded_pred_slice)

        self.slice_patch_size = self.get_patch_size(self.loaded_mask_slice)
        if self.loaded_target is not None:
            self.loaded_target_slice = self.loaded_target[slice_index]
            self.loaded_target_slice = harray.scale_minmax(self.loaded_target_slice)
            self.loaded_target_slice = img_as_ubyte(self.loaded_target_slice)
            # self.loaded_target_slice = clahe.apply(self.loaded_target_slice)
        #
        # # Equalizing the images...
        # equalize_obj = hplotc.ImageIntensityEqualizer(reference_image=self.loaded_image_slice, image_list=[self.loaded_pred_slice],
        #                                               patch_width=self.patch_size,
        #                                               dynamic_thresholding=True)
        # temp_images = equalize_obj.correct_image_list()
        # self.loaded_pred_slice = np.array(temp_images[0])
        # self.loaded_image_slice[self.loaded_image_slice > equalize_obj.vmax_ref] = equalize_obj.vmax_ref
        # self.loaded_image_slice = harray.scale_minmax(self.loaded_image_slice)

        # Adding this... again...
        # Wil ik dit echt....? -> Geen idee voor nu 04/07/2022
        # import cv2
        # nx, ny = self.loaded_mask_slice.shape
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nx // 32, ny // 32))
        # self.loaded_image_slice = clahe.apply(self.loaded_image_slice)
        # self.loaded_pred_slice = clahe.apply(self.loaded_pred_slice)
        # if self.dtarget is not None:
        #     self.loaded_target_slice = clahe.apply(self.loaded_target_slice)
        # Weg gehaald voor nu... 12/07/2022

    @staticmethod
    def load_mask(x):
        loaded_mask = hmisc.load_array(x)
        if loaded_mask.ndim == 2:
            loaded_mask = loaded_mask[None]
        loaded_mask = skimage.img_as_bool(loaded_mask).astype(int)
        if 'nii' in hmisc.get_ext(x):
            loaded_mask = loaded_mask.T[:, ::-1, ::-1]
        return loaded_mask

    def run_features(self, debug=False):
        # Do a complete run...fixed to to a patch-based evaluation
        glcm_rel = []
        glcm_input = []
        glcm_pred = []
        coef_var_rel = []
        coef_var_input = []
        coef_var_pred = []
        slice_list = []
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            self.print_image_info()
            # Somewhere we suddenly assume a loaded_mask to exist....
            # It should work without as well...
            if self.mid_slice:
                range_of_slices = [self.n_slices // 2]
            else:
                # It is required to make a list of these since a range cant be serialized by json
                if self.n_slices < 20:
                    range_of_slices = list(range(self.n_slices))
                else:
                    # whyyy.... Moved away from the range(10, 20) to this thing below
                    # This reduces the calculations SOMEWHAT
                    # And still leaves me with a middle slice..
                    # BWAH
                    lower_limit = int(0.10 * self.n_slices)
                    upper_limit = int(0.90 * self.n_slices)
                    range_of_slices = list(range(lower_limit, upper_limit))

            print(f'Running slice')
            for slice_index in range_of_slices:
                print(f'Running slice {slice_index} / {self.n_slices}', end='\r')
                self.set_slice(slice_index)
                if self.debug:
                    self.print_slice_info()
                self.loaded_mask_slice = harray.shrink_image(self.loaded_mask_slice, self.shrink_pixels)
                self.loaded_image_slice = np.ma.masked_array(self.loaded_image_slice, mask=1 - self.loaded_mask_slice)
                self.loaded_pred_slice = np.ma.masked_array(self.loaded_pred_slice, mask=1 - self.loaded_mask_slice)
                center_mask = harray.create_random_center_mask(self.loaded_image_slice.shape, random=False)

                mean_pred = np.mean(self.loaded_pred_slice[center_mask == 1])
                mean_input = np.mean(self.loaded_image_slice[center_mask == 1])
                pred_to_input_scaling = mean_input / mean_pred
                # Mean intensity value is now similar between prediction and target
                # Why.... multply with this?
                self.loaded_image_slice = self.loaded_image_slice * 1.
                self.loaded_pred_slice = self.loaded_pred_slice * pred_to_input_scaling

                self.loaded_image_slice = self._prep_image(self.loaded_image_slice)
                self.loaded_pred_slice = self._prep_image(self.loaded_pred_slice)

                # Get GLCM features
                # Cropping is needed to avoid pathces without much info...
                # 04-7-22: removed this because we are using masked valued
                # temp_pred_slice, temp_mask_slice = harray.get_crop(self.loaded_pred_slice, self.loaded_mask_slice)
                # temp_image_slice, _ = harray.get_crop(self.loaded_image_slice, self.loaded_mask_slice)
                rel_temp_dict, pred_temp_dict, input_temp_dict = self.get_glcm_slice(self.loaded_pred_slice, self.loaded_image_slice, patch_size=self.slice_patch_size)
                glcm_rel.append(rel_temp_dict)
                glcm_input.append(input_temp_dict)
                glcm_pred.append(pred_temp_dict)
                # Get relative COV features
                coeff_var_input_value = np.std(self.loaded_image_slice) / np.mean(self.loaded_image_slice)
                coeff_var_pred_value = np.std(self.loaded_pred_slice) / np.mean(self.loaded_pred_slice)
                coeff_var_rel_value = (coeff_var_pred_value - coeff_var_input_value) / coeff_var_input_value
                if debug:
                    logger.info(f"Relative GLCM dict {rel_temp_dict}")
                    logger.info(f"Relative COefv {coeff_var_rel_value}")
                coef_var_rel.append(coeff_var_rel_value)
                coef_var_input.append(coeff_var_input_value)
                coef_var_pred.append(coeff_var_pred_value)
                slice_list.append(range_of_slices)
        return glcm_rel, glcm_input, glcm_pred, coef_var_rel, coef_var_input, coef_var_pred, slice_list

    def run_features_target(self):
        # Do a complete run...fixed to to a patch-based evaluation
        glcm_rel = []
        glcm_target = []
        coefv_target = []
        coefv_target_rel = []
        RMSE_list = []
        WSS_distance = []
        SSIM_list = []
        slice_list = []
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            self.print_image_info()
            # Somewhere we suddenly assume a loaded_mask to exist....
            # It should work without as well...
            if self.mid_slice:
                range_of_slices = [self.n_slices // 2]
            else:
                if self.n_slices < 20:
                    range_of_slices = list(range(self.n_slices))
                else:
                    range_of_slices = list(range(10, 20))

            for slice_index in range_of_slices:
                print(f'Running slice {slice_index} / {self.n_slices}', end='\r')
                self.set_slice(slice_index)
                # Shrink the mask... and check it out...?
                self.loaded_mask_slice = harray.shrink_image(self.loaded_mask_slice, self.shrink_pixels)
                # ...
                self.loaded_target_slice = np.ma.masked_array(self.loaded_target_slice, mask=1 - self.loaded_mask_slice)
                self.loaded_image_slice = np.ma.masked_array(self.loaded_image_slice, mask=1 - self.loaded_mask_slice)
                self.loaded_pred_slice = np.ma.masked_array(self.loaded_pred_slice, mask=1 - self.loaded_mask_slice)

                center_mask = harray.create_random_center_mask(self.loaded_image_slice.shape, random=False)
                # Make sure target and input image are similar in scaling
                # Create a mask in the center from which we get values..
                mean_pred = np.mean(self.loaded_pred_slice[center_mask==1])
                mean_input = np.mean(self.loaded_image_slice[center_mask==1])
                mean_target = np.mean(self.loaded_target_slice[center_mask==1])
                pred_to_target_scaling = mean_target / mean_pred
                input_to_target_scaling = mean_target / mean_input
                # Mean intensity value is now similar between prediction and target
                self.loaded_image_slice = self.loaded_image_slice * input_to_target_scaling
                self.loaded_pred_slice = self.loaded_pred_slice * pred_to_target_scaling
                self.loaded_target_slice = self.loaded_target_slice * 1.

                # Convert to uint8-type again, while tresholding some values
                self.loaded_image_slice = self._prep_image(self.loaded_image_slice)
                self.loaded_pred_slice = self._prep_image(self.loaded_pred_slice)
                self.loaded_target_slice = self._prep_image(self.loaded_target_slice)

                # if self.debug:
                #     print("Image characteristics...")
                #     print("Loaded image slice ", harray.get_minmeanmediammax(self.loaded_image_slice), self.loaded_image_slice.dtype)
                #     print("Loaded pred slice ", harray.get_minmeanmediammax(self.loaded_pred_slice), self.loaded_pred_slice.dtype)
                #     print("Loaded target slice ", harray.get_minmeanmediammax(self.loaded_target_slice), self.loaded_target_slice.dtype)

                # Calculate histograms
                slice_hist_target, _ = np.histogram(self.loaded_target_slice.ravel(), bins=256, range=(0, 255),
                                                    density=True)
                slice_hist_pred, _ = np.histogram(self.loaded_pred_slice.ravel(), bins=256, range=(0, 255),
                                                  density=True)

                # Calculate all the features..
                rel_temp_dict, target_temp_dict, input_temp_dict = self.get_glcm_slice(self.loaded_target_slice, self.loaded_image_slice, patch_size=self.slice_patch_size)
                coeff_var_target_value = np.std(self.loaded_target_slice) / np.mean(self.loaded_target_slice)
                coeff_var_input_value = np.std(self.loaded_image_slice) / np.mean(self.loaded_image_slice)
                coeff_var_rel_value = (coeff_var_target_value - coeff_var_input_value) / coeff_var_input_value

                rmse_value = np.sqrt(np.mean((self.loaded_pred_slice - self.loaded_target_slice) ** 2))
                wss_value = scipy.stats.wasserstein_distance(slice_hist_pred, slice_hist_target)
                ssim_target_pred = skimage.metrics.structural_similarity(self.loaded_pred_slice, self.loaded_target_slice, data_range=255)

                if self.debug:
                    print("Scaling values ", pred_to_target_scaling)
                    print("Min/Mean/Median/Max mask ", harray.get_minmeanmediammax(self.loaded_mask_slice))
                    print("Min/Mean/Median/Max image ", harray.get_minmeanmediammax(self.loaded_image_slice))
                    print("Min/Mean/Median/Max target ", harray.get_minmeanmediammax(self.loaded_target_slice))
                    print("Min/Mean/Median/Max pred ", harray.get_minmeanmediammax(self.loaded_pred_slice))
                    print("Wasserstein distance ", wss_value)
                    print("SSIM ", ssim_target_pred)
                    print("RMSE ", rmse_value)
                coefv_target.append(coeff_var_target_value)
                coefv_target_rel.append(coeff_var_rel_value)
                glcm_target.append(target_temp_dict)
                glcm_rel.append(rel_temp_dict)
                RMSE_list.append(rmse_value)
                SSIM_list.append(ssim_target_pred)
                WSS_distance.append(wss_value)
                slice_list.append(range_of_slices)
        return coefv_target_rel, coefv_target, glcm_rel, glcm_target, RMSE_list, SSIM_list, WSS_distance, slice_list

    @staticmethod
    def _prep_image(x):
        # Forces an image to become a uint8 (0, 255)
        # This is used to convert a float image near the range of (0, 255) to become a uint8 image
        x = np.copy(x)
        x[x > 255] = 255
        x = x.astype(np.uint8)
        return x

    def run_target_input_features(self):
        # Do a complete run...fixed to to a patch-based evaluation
        RMSE_list = []
        WSS_distance = []
        SSIM_list = []
        slice_list = []
        for i_index, i_file in enumerate(self.file_list):
            print('Running file ', i_file)
            self.load_file(file_index=i_index)
            self.print_image_info()
            # Somewhere we suddenly assume a loaded_mask to exist....
            # It should work without as well...
            if self.mid_slice:
                range_of_slices = [self.n_slices // 2]
            else:
                if self.n_slices < 20:
                    range_of_slices = list(range(self.n_slices))
                else:
                    range_of_slices = list(range(10, 20))

            for slice_index in range_of_slices:
                print(f'Running slice {slice_index} / {self.n_slices}', end='\r')
                self.set_slice(slice_index)
                # Shrink the mask... and check it out...?
                self.loaded_mask_slice = harray.shrink_image(self.loaded_mask_slice, self.shrink_pixels)
                # ...
                self.loaded_target_slice = np.ma.masked_array(self.loaded_target_slice, mask=1 - self.loaded_mask_slice)
                self.loaded_image_slice = np.ma.masked_array(self.loaded_image_slice, mask=1 - self.loaded_mask_slice)

                center_mask = harray.create_random_center_mask(self.loaded_image_slice.shape, random=False)
                # Make sure target and input image are similar in scaling
                # Create a mask in the center from which we get values..
                mean_input = np.mean(self.loaded_image_slice[center_mask==1])
                mean_target = np.mean(self.loaded_target_slice[center_mask==1])
                input_to_target_scaling = mean_target / mean_input
                # Mean intensity value is now similar between prediction and target
                self.loaded_image_slice = self.loaded_image_slice * input_to_target_scaling
                self.loaded_target_slice = self.loaded_target_slice * 1.
                #
                # Convert to uint8-type again, while tresholding some values
                self.loaded_image_slice = self._prep_image(self.loaded_image_slice)
                self.loaded_target_slice = self._prep_image(self.loaded_target_slice)


                # Calculate histograms
                slice_hist_target, _ = np.histogram(self.loaded_target_slice.ravel(), bins=256, range=(0, 255),
                                                    density=True)
                slice_hist_image, _ = np.histogram(self.loaded_image_slice.ravel(), bins=256, range=(0, 255),
                                                  density=True)

                rmse_value = np.sqrt(np.mean((self.loaded_image_slice - self.loaded_target_slice) ** 2))
                wss_value = scipy.stats.wasserstein_distance(slice_hist_image, slice_hist_target)
                ssim_target_pred = skimage.metrics.structural_similarity(self.loaded_image_slice,
                                                                         self.loaded_target_slice,
                                                                         data_range=255)

                if self.debug:
                    print("Min/Mean/Median/Max mask ", harray.get_minmeanmediammax(self.loaded_mask_slice))
                    print("Min/Mean/Median/Max image ", harray.get_minmeanmediammax(self.loaded_image_slice))
                    print("Min/Mean/Median/Max target ", harray.get_minmeanmediammax(self.loaded_target_slice))
                    print("Min/Mean/Median/Max pred ", harray.get_minmeanmediammax(self.loaded_pred_slice))
                    print("Wasserstein distance ", wss_value)
                    print("SSIM ", ssim_target_pred)
                    print("RMSE ", rmse_value)

                RMSE_list.append(rmse_value)
                SSIM_list.append(ssim_target_pred)
                WSS_distance.append(wss_value)
                slice_list.append(range_of_slices)
        return RMSE_list, SSIM_list, WSS_distance, slice_list

    def get_patch_size(self, x):
        if self.patch_size is None:
            patch_size = min(x.shape) // 3
        elif self.patch_size == 'max':
            patch_size = min(x.shape)
        else:
            patch_size = self.patch_size
        return patch_size

    def get_glcm_slice(self, image_a, image_b, patch_size):
        logger.info("Get GLCM slice ... ")
        logger.info(f"image A shape {image_a.shape} ")
        logger.info(f"image B shape {image_b.shape} ")
        logger.info(f"patch size {patch_size} ")
        glcm_obj_a = homog_metric.get_glcm_patch_object(image_a, patch_size=patch_size, glcm_dist=self.glcm_dist)
        glcm_obj_b = homog_metric.get_glcm_patch_object(image_b, patch_size=patch_size, glcm_dist=self.glcm_dist)
        feature_dict_rel = {}
        feature_dict_a = {}
        feature_dict_b = {}
        n_patches = float(len(glcm_obj_a))
        counter = 0
        for patch_obj_a, patch_obj_b in zip(glcm_obj_a, glcm_obj_b):
            counter += 1
            if counter % int(0.1 * n_patches) == 0:
                print(f'GLCM patch counter : {counter}', end='\r')
            for i_feature in self.feature_keys:
                _ = feature_dict_rel.setdefault(i_feature, 0)
                _ = feature_dict_a.setdefault(i_feature, 0)
                _ = feature_dict_b.setdefault(i_feature, 0)
                feature_value_a = skimage.feature.graycoprops(patch_obj_a, i_feature)
                feature_value_b = skimage.feature.graycoprops(patch_obj_b, i_feature)
                rel_change = (feature_value_a - feature_value_b) / feature_value_b
                feature_dict_rel[i_feature] += np.mean(rel_change) / n_patches
                feature_dict_a[i_feature] += np.mean(feature_value_a) / n_patches
                feature_dict_b[i_feature] += np.mean(feature_value_b) / n_patches

        return feature_dict_rel, feature_dict_a, feature_dict_b
