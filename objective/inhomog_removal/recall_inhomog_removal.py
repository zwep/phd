"""
Extend the Recall Base class with some stuff of my own
"""

import time
from skimage.util import img_as_ubyte, img_as_int, img_as_uint
import scipy.spatial.distance
import scipy.stats
import argparse
import os
import torch

import helper.array_transf
import helper.misc as hmisc
import helper.array_transf as harray
import objective.inhomog_removal.executor_inhomog_removal as executor
from objective.recall_base import RecallBase
import matplotlib.pyplot as plt
import numpy as np
import json
import biasfield_algorithms.N4ITK as model_n4itk
import itertools
import helper.metric as hmetric
import helper.plot_class as hplotc
import sporco.metric
import getpass


class RecallInhomog(RecallBase):
    def test_model_additional_metric(self, modelrun_obj):
        prediction_mode = modelrun_obj.config_param['data']['prediction_mode']
        transform_type = modelrun_obj.config_param['data']['transform_type']
        temp_dict = {'prediction_mode': prediction_mode, 'transform_type': transform_type}

        modelrun_obj.model_obj.eval()  # IMPORTANT

        additional_metric_list = []
        with torch.no_grad():  # IMPORTANT
            for container in modelrun_obj.test_loader:
                X, y, y_pred, mask = self.make_prediction(modelrun_obj, modelrun_obj.model_obj, container)
                # This is needed if we every want to calculate more metrics on the test data..
                # This was the case for the inhomog removal thing.
                output_cont = self.postproc_container(X, y, y_pred, mask, **temp_dict)
                orig_input, corrected_image, target, bias_field, mask = output_cont

                temp = self.additional_metric(orig_input, corrected_image, target, bias_field, mask)
                additional_metric_list.append(temp)

        return additional_metric_list

    def additional_metric(self, orig_input, corrected_image, target, bias_field, mask):
        # Here we want to calculate.... addition metrics on the test data set ...
        t0 = time.time()
        print('Starting debug additional metric', time.asctime())

        if self.debug:
            print('Input min max', orig_input.min(), orig_input.max())
            print('Corrected input min max', corrected_image.min(), corrected_image.max())
            print('Target iamge min max', target.min(), target.max())
            # print('n4itk cor min max', corrected_n4itk.min(), corrected_n4itk.max())

        compare_array = [orig_input, target, corrected_image]
        compare_labels = ['input', 'target', 'corrected_pred']
        temp_dict = self.calculate_metric(compare_array=compare_array, mask=mask, compare_labels=compare_labels,
                                          reference_image=compare_array[1])
        temp_dict_alone = self.calculate_metric_stand_alone(compare_array=compare_array, mask=mask, compare_labels=compare_labels)
        hmisc.update_nested(temp_dict, temp_dict_alone)

        print('Finished', time.asctime(), time.time() - t0)
        return temp_dict

    def calculate_metric(self, compare_array, compare_labels, mask, reference_image):
        # Fixed combinations to reduce the amount of computation
        # This compares the following
        #   input - target
        #   input - pred
        #   pred - target
        # All metrics should be symmetrical
        # Now all I need is predictions with n4itk and their metrics....

        # Normalize each image to 0..1
        compare_array = [harray.scale_minmax(x) for x in compare_array]
        image_ref = harray.scale_minmax(reference_image)
        image_ref_256 = img_as_ubyte(image_ref)
        hist_ref, _ = np.histogram(img_as_ubyte(image_ref_256[mask == 1]).ravel(), bins=256, range=(0, 255), density=True)

        n_max_power = 8
        temp_dict = {}
        for i_label, i_image in zip(compare_labels, compare_array):
            image_pred_256 = img_as_ubyte(i_image)
            hist_pred, _ = np.histogram(image_pred_256[mask == 1].ravel(), bins=256, range=(0, 255), density=True)
            temp_dict.setdefault(i_label, {})

            temp_wasserstein_distance = scipy.stats.wasserstein_distance(hist_pred, hist_ref)
            temp_jensen_shannon = scipy.spatial.distance.jensenshannon(hist_pred,  hist_ref)

            hpsi_value = hmetric.haar_psi_numpy(image_pred_256,  image_ref_256)[0]
            fsim_value = hmetric.fsim(image_pred_256[:, :, None], image_ref_256[:, :, None])
            ssim_value = hmetric.ssim(image_pred_256[:, :, None], image_ref_256[:, :, None], max_p=None)
            mae_value = sporco.metric.mae(vref=reference_image, vcmp=i_image)
            snr_value = sporco.metric.snr(vref=reference_image, vcmp=i_image)
            # PAMSE paper
            # http://www4.comp.polyu.edu.hk/~cslzhang/paper/conf/ICCV13/PAMSE.pdf
            pamse_value = sporco.metric.pamse(vref=reference_image, vcmp=i_image)
            # GMSD paper
            # https://arxiv.org/pdf/1308.3052.pdf
            gmsd_value = sporco.metric.gmsd(vref=reference_image, vcmp=i_image)

            temp_dict[i_label]['wasserstein'] = float(temp_wasserstein_distance)
            temp_dict[i_label]['jensen_shannon'] = float(temp_jensen_shannon)

            temp_dict[i_label]['hpsi'] = float(hpsi_value)
            temp_dict[i_label]['fsim'] = float(fsim_value)
            temp_dict[i_label]['ssim'] = float(ssim_value)
            temp_dict[i_label]['dssim'] = (1 - float(ssim_value)) / 2
            temp_dict[i_label]['mae'] = float(mae_value)
            temp_dict[i_label]['snr'] = float(snr_value)
            temp_dict[i_label]['pamse'] = float(pamse_value)
            temp_dict[i_label]['gmsd'] = float(gmsd_value)

            temp_dict[i_label]['hi'] = [float(x) for x in self.get_mean_power(i_image, mask=mask, max_power=n_max_power)]
            temp_dict[i_label]['hi'] = [float(x) for x in self.get_mean_power(i_image, mask=mask, max_power=n_max_power)]

        return temp_dict

    def calculate_metric_stand_alone(self, compare_array, compare_labels, mask):
        # Normalize each image to 0..1
        compare_array = [harray.scale_minmax(x) for x in compare_array]

        temp_dict = {}
        for i_label, i_image in zip(compare_labels, compare_array):
            temp_dict.setdefault(i_label, {})
            counts, bins = np.histogram(i_image.ravel(), range=(0 + 1e-8, 1-1e-8), bins=256, density=True)

            temp_dict[i_label]['entropy'] = float(scipy.stats.entropy(pk=counts, base=2))
            temp_dict[i_label]['cov'] = float(hmetric.coefficient_of_variation(i_image[mask == 1]))
            temp_dict[i_label]['std'] = float(i_image[mask == 1].std())
            temp_dict[i_label]['distortion'] = float(np.abs(i_image[mask == 1] - i_image[mask == 1].mean()).mean())

        return temp_dict

    @staticmethod
    def get_abs_output(y_pred):

        # This processes the input and prediction to a nice corrected image
        # Used to always get the image out..
        if y_pred.shape[1] > 1:
            # Here we are dealing with.. expansion stuff
            output_cpx = y_pred.numpy()[0][0] + 1j * y_pred.numpy()[0][1]
            # target_cpx = target.numpy()[0][0] + 1j * target.numpy()[0][1]
            # Here we take the ABS of the output..
            output_abs = np.abs(output_cpx)
            # target_abs = np.abs(target_cpx)
        else:
            # Output is either abs or real.. any case.. it is fine..
            output_abs = y_pred.numpy()[0][0]
            # target_abs = target.numpy()[0][0]

        return output_abs

    @staticmethod
    def postproc_rho(abs_sum_input, output_abs, mask):
        # Screw all those dificulties...
        corrected_image = output_abs * mask
        # corrected_image = 1 + (corrected_image - corrected_image.mean(axis=(-2, -1), keepdims=True)) / np.abs(
        #     corrected_image).std(axis=(-2, -1), keepdims=True)
        bias_field = abs_sum_input / corrected_image
        bias_field = helper.array_transf.correct_inf_nan(bias_field)
        # bias_field_smoothed_adaptive = harray.smooth_image(bias_field, n_kernel=bias_field.shape[0] // 32,
        #                                                    conv_boundary='wrap')
        # bias_field_smoothed_adaptive = 1 + (
        #         bias_field_smoothed_adaptive - bias_field_smoothed_adaptive.mean(axis=(-2, -1),
        #                                                                          keepdims=True)) / np.abs(
        #     bias_field_smoothed_adaptive).std(axis=(-2, -1), keepdims=True)
        # corrected_image = abs_sum_input / bias_field_smoothed_adaptive
        # corrected_image = hmisc.correct_inf_nan(corrected_image) * mask

        return bias_field, corrected_image

    @staticmethod
    def postproc_biasfield(abs_sum_input, output_abs, mask):
        bias_field = output_abs * mask
        # Scaled to N(1, std) so that the division is better...
        # Not doing this anymore for the test split...
        # bias_field = 1 + (bias_field - bias_field.mean(axis=(-2, -1), keepdims=True)) / np.abs(bias_field).std(axis=(-2, -1), keepdims=True)
        corrected_image = abs_sum_input / bias_field
        corrected_image = helper.array_transf.correct_inf_nan(corrected_image) * mask

        return bias_field, corrected_image

    def postproc_container(self, X, y, y_pred, mask, **kwargs):
        prediction_mode = kwargs.get('prediction_mode', None)
        if prediction_mode is None:
            print('Error, no prediction mode is given/found', kwargs)

        transform_type = kwargs.get('transform_type', None)
        if transform_type is None:
            print('Error, no transform type is given/found', kwargs)

        abs_sum_input = self.postproc_input(X, transform_type)
        # Omdat in 'get_stuff' de target_type op 'rho' staat zal dit altijd een
        # homogeen plaatje zijn.

        target = y.numpy()[0][0]
        mask = mask.numpy()[0][0]

        output_abs = self.get_abs_output(y_pred)

        # Correct for output if we have any change in size...
        if output_abs.shape != abs_sum_input.shape:
            print('Correcting for output shape mismatch')
            import skimage.transform as sktransform
            output_abs = sktransform.resize(output_abs, abs_sum_input.shape, preserve_range=True)

        if prediction_mode == 'biasfield':
            bias_field, corrected_image = self.postproc_biasfield(abs_sum_input, output_abs, mask)
        elif prediction_mode == 'rho':
            # This used to be different I guess for rho/biasfield predictions
            bias_field, corrected_image = self.postproc_rho(abs_sum_input, output_abs, mask)
        else:
            bias_field = None
            corrected_image = None

        # Dit is blijkbaar wat je wilt....
        scale_factor = harray.get_proper_scaled(corrected_image, patch_shape=tuple(np.array(corrected_image.shape) // 30))
        # Dit gaat lekker. NU is ie tussen 0..1 geschaald en gaat alles goed/beter
        corrected_image = corrected_image / scale_factor
        # DIt is er bij gedaan.. Misschien werkt dit wel beter?
        corrected_image = corrected_image.astype(np.float32)
        corrected_image[corrected_image > 1] = 1
        corrected_image[corrected_image < 0] = 0

        return abs_sum_input, corrected_image, target, bias_field, mask

    @staticmethod
    def get_mean_power(x, mask, max_power):
        # Investigate the output... based on subcutenous fat
        power_mean_value = [(x[mask == 1] ** n_power).mean() for n_power in range(max_power)]
        return power_mean_value

    @staticmethod
    def plot_power_list(x, title='Mean intensity', labels=None, save=True):
        power_array = np.array(x)
        power_colors = ['k', 'r', 'b']
        if labels is None:
            power_labels = ['input', 'real input', 'complex input']
        else:
            power_labels = labels

        power_colors = power_colors[:len(power_labels)]

        fig, ax = plt.subplots()
        for i, color in enumerate(power_colors):
            # This get the input array, output of single or output of multi
            x = power_array[:, i, :]
#            x_95 = np.percentile(x, q=95, axis=0)
 #           x_05 = np.percentile(x, q=5, axis=0)
            x_50 = np.percentile(x, q=50, axis=0)
            ax.plot(x_50, color, label=power_labels[i])

        plt.legend()
        plt.title(title)
        plt.xlabel('Power')
        plt.ylabel('Mean intensity')
        #    ax.plot(x_05, color + '--', alpha=0.5)
        #    ax.plot(x_95, color + '--', alpha=0.5)
        #    ax.fill_between(range(n_power_analysis), x_05, x_95, facecolor=color, alpha=0.1)
        if save:
            fig.savefig('/local_scratch/sharreve/test_add_metric.png')
            print('Stored in location /local_scratch/sharreve/test_add_metric.png')

    def get_and_store_metric(self, cycle_all=False, dest_dir=None):
        for config_name, i_config in self.mult_dict.items():
            print('Dealing with config ', i_config)
            full_model_path = os.path.join(self.model_path, config_name)
            # Changing target type to 'rho'
            # and adding prediction mode as way of post processing
            target_type = i_config['data']['target_type']
            i_config['data']['prediction_mode'] = target_type
            # This way we definitely have the homogeneous image as target image..
            i_config['data']['target_type'] = 'rho'
            # With this on we are going to cycle through all the test items..
            i_config['data']['cycle_all'] = cycle_all
            # We dont want to mess up the order of the files...
            # This way the first file will be the same over all the models..
            i_config['data']['shuffle'] = False

            print('Getting model object')
            hmisc.print_dict(i_config)
            i_config['dir']['doutput'] = full_model_path

            modelrun_obj = self.get_model_object(i_config)
            print('Calculating test model with additional metric')
            metric_result = self.test_model_additional_metric(modelrun_obj)

            if dest_dir is None:
                additional_metric_file = os.path.join(full_model_path, 'additional_metric.json')
            else:
                additional_metric_file = os.path.join(dest_dir, 'additional_metric.json')

            json_ser_obj = json.dumps(metric_result)
            with open(additional_metric_file, 'w') as f:
                f.write(json_ser_obj)

    def get_and_store_prediction(self, dest_dir=None):
        # If we dont provide a destination dir... we use the model directory itself

        for config_name, i_config in self.mult_dict.items():
            print('Dealing with config ', i_config)
            full_model_path = os.path.join(self.model_path, config_name)

            # Changing target type to 'rho'
            # and adding prediction mode as way of post processing
            target_type = i_config['data']['target_type']
            i_config['data']['prediction_mode'] = target_type
            # This way we definitely have the homogeneous image as target image..
            i_config['data']['target_type'] = 'rho'
            # We are NOT going to cycle over all, we just want 32 predictions [False]
            # We are going to cycle over all [True]
            i_config['data']['cycle_all'] = True
            # We just want to take the CENTER slice.
            # Not anymore... we want to get ALL the slices...
            i_config['data']['center_slice'] = False
            # We dont want to mess up the order of the files...
            i_config['data']['shuffle'] = False
            # We dont want to resize stuff..
            i_config['data']['transform_resize'] = False

            print('Getting model object')
            hmisc.print_dict(i_config)
            if dest_dir is None:
                storage_path = os.path.join(full_model_path, 'storage_results')
            else:
                storage_path = os.path.join(dest_dir, config_name, 'storage_results')

            if not os.path.isdir(storage_path):
                os.makedirs(storage_path)

            i_config['dir']['doutput'] = full_model_path
            modelrun_obj = self.get_model_object(i_config)
            prediction_mode = modelrun_obj.config_param['data']['prediction_mode']
            transform_type = modelrun_obj.config_param['data']['transform_type']
            n_bins = modelrun_obj.config_param['data'].get('bins_expansion', False)

            temp_dict = {'prediction_mode': prediction_mode, 'transform_type': transform_type, 'n_bins': n_bins}

            modelrun_obj.model_obj.eval()  # IMPORTANT
            counter = 0
            with torch.no_grad():  # IMPORTANT
                for container in modelrun_obj.test_loader:
                    print('Starting with prediction number ', counter)
                    sel_slice = container['sel_slice'][0]
                    sel_file = os.path.basename(container['file_name'][0])
                    sel_file, _ = os.path.splitext(sel_file)
                    print('File name', sel_file)
                    print('Slice number', sel_slice)
                    target_file = os.path.join(storage_path, f'target_{sel_file}_{sel_slice}.npy')
                    if os.path.isfile(target_file):
                        continue
                    else:
                        X, y, y_pred, mask = self.make_prediction(modelrun_obj, container)
                        # This is needed if we every want to calculate more metrics on the test data..
                        # This was the case for the inhomog removal thing.
                        output_cont = self.postproc_container(X, y, y_pred, mask, **temp_dict)
                        orig_input, corrected_image, target, bias_field, mask = output_cont
                        orig_input = self.convert_to_ubyte(orig_input)
                        corrected_image = self.convert_to_ubyte(corrected_image)
                        target = self.convert_to_ubyte(target)

                        np.save(os.path.join(storage_path, f'input_{sel_file}_{sel_slice}.npy'), orig_input)
                        np.save(os.path.join(storage_path, f'pred_{sel_file}_{sel_slice}.npy'), corrected_image)
                        np.save(os.path.join(storage_path, f'target_{sel_file}_{sel_slice}.npy'), target)
                    counter += 1

    @staticmethod
    def convert_to_ubyte(x):
        return img_as_ubyte(harray.scale_minmax(x))

    def get_and_store_prediction_debug(self):
        # With this we can re-do some of the caluclations and check how we can get the
        # best post porcessed output... because that is currently lacking, and I think it is mostly visual.
        for config_name, i_config in self.mult_dict.items():
            print('Dealing with config ', i_config)
            full_model_path = os.path.join(self.model_path, config_name)
            storage_path = os.path.join(full_model_path, 'storage_results')
            if not os.path.isdir(storage_path):
                os.makedirs(storage_path)

            # Changing target type to 'rho'
            # and adding prediction mode as way of post processing
            target_type = i_config['data']['target_type']
            i_config['data']['prediction_mode'] = target_type
            # This way we definitely have the homogeneous image as target image..
            i_config['data']['target_type'] = 'rho'
            # We are NOT going to cycle over all, we just want 32 predictions
            i_config['data']['cycle_all'] = False
            # We just want to take the CENTER slice.
            i_config['data']['center_slice'] = True
            # We dont want to mess up the order of the files...
            i_config['data']['shuffle'] = False

            print('Getting model object')
            hmisc.print_dict(i_config)
            i_config['dir']['doutput'] = full_model_path
            modelrun_obj = self.get_model_object(i_config)
            prediction_mode = modelrun_obj.config_param['data']['prediction_mode']
            transform_type = modelrun_obj.config_param['data']['transform_type']
            n_bins = modelrun_obj.config_param['data'].get('bins_expansion', False)

            temp_dict = {'prediction_mode': prediction_mode, 'transform_type': transform_type, 'n_bins': n_bins}

            modelrun_obj.model_obj.eval()  # IMPORTANT
            counter = 0
            with torch.no_grad():  # IMPORTANT
                for container in modelrun_obj.test_loader:
                    print('Starting with prediction number ', counter)
                    X, y, y_pred, mask = self.make_prediction(modelrun_obj, container)
                    break

            return X, y, y_pred, mask, temp_dict
