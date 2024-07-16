# encoding: utf-8

import data_generator.UndersampledRecon as data_gen_recon
import data_generator.InhomogRemoval as data_gen_inhomog
import data_generator.Default as data_gen_default
import data_generator.UndersampledRecon as data_gen_undersampled
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase

from objective.executor_cycleGAN import ExecutorCycleGAN
import torch
import numpy as np
import helper.plot_class as hplotc
import os
import helper.array_transf as harray


class DecisionMaker(DecisionmakerBase):
    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        if 'cyclegan' == model_choice.lower():
            if self.debug:
                print('You have chosen a Cycle GAN')
            sel_executor = ExecutorInhomogRemovalCycleGAN
        elif 'gan' == model_choice.lower():
            if self.debug:
                print('You have chosen a regular GAN')
            sel_executor = ExecutorInhomogRemovalGAN
        elif 'regular' == model_choice.lower():
            if self.debug:
                print('You have chosen a regular model', model_choice.lower())
            sel_executor = ExecutorInhomogRemoval
        else:
            print('No model choice is found.. used: ', model_choice.lower())
            sel_executor = None

        return sel_executor(config_file=self.config_param, **self.kwargs)


class ExecutorInhomogRemoval(ExecutorRegular):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "inhomogh5":
            data_gen_sel = data_gen_inhomog.DataGeneratorInhomogRemovalH5
        elif dg_choice == "inhomog_cardiac":
            data_gen_sel = data_gen_inhomog.DataGeneratorCardiac
        elif dg_choice == "noise":
            data_gen_sel = data_gen_default.DataGeneratorNoise
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator  # ppff simple fucntion

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj

    def _correct_bias_field_prediction(self, input, to_be_corrected):
        # This function is only there to re-create biasfield predictions
        if self.debug:
            print('Calculating corrected bias field ')
            print('Input: ', input.shape)
            print('to_be_corrected: ', to_be_corrected.shape)

        temp_tensor = torch.as_tensor(input)
        temp_tensor = temp_tensor.to(self.device)
        bias_field = self.model_obj(temp_tensor)
        if self.debug:
            print('Model output shape')
            print('bias_field ', bias_field.shape)

        bias_field = np.array(bias_field[0][0].detach().cpu().numpy())
        corrected_image = to_be_corrected / bias_field
        return corrected_image

    @staticmethod
    def _hack_prep_input(x):
        # Hack-ish manier om iets te coden.. niet er gnetjes heh
        n_y, n_x = x.shape[-2:]
        x_inputed = harray.to_stacked(x, cpx_type='cartesian', stack_ax=0)
        x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
        A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()
        return A_tensor

    def save_model(self, plot_name='final'):
        # We want to plot... the biasfield stuff..
        # But also want to test on real data....
        # We do the initial thing.
        super().save_model(plot_name=plot_name)
        #
        # print('\t Performing additional plotting')
        # # And some additional plotting..
        # # This is all pretty improvised and NOT nicely written
        #
        # doutput = self.config_param['dir']['doutput']
        # output_path = os.path.join(doutput, f"compare_stuff.jpg")
        # main_path = '/local_scratch/sharreve/mri_data/measured_prostate'
        # measured_data_dir = os.path.join(main_path, 't2w/v9_15072020_1756533_21_3_t2wV4.npy')
        # mask_data_dir = os.path.join(main_path, 'body_mask/v9_15072020_1756533_21_3_t2wV4.npy')
        #
        # # Only do this when we are dealing with bias field predictions
        # target_type = self.config_param['data']['target_type']
        #
        # if (target_type == 'biasfield') or (target_type == 'expansion'):
        #     input_test_set, target_test_set, pred_test_set, plot_augmentation = self.get_image_prediction()
        #     transform_type_input = self.config_param['data']['transform_type']
        #     # transform_type_target = self.config_param['data']['transform_type_target']
        #
        #     # First fix the bias field corrected image on the test set
        #     # The input can be.. either a 8-channel complex valued coil
        #     # Or something totally different..
        #     # We want to apply it to the interference image...
        #     print('Input shape ')
        #     input_test_set = self.train_loader.dataset.transform_complex(input_test_set, transform_type=transform_type_input)
        #
        #     corrected_input_test_set = pred_test_set / input_test_set
        #     x_input = np.load(measured_data_dir)
        #     x_input = self.train_loader.dataset.transform_complex(x_input, transform_type=transform_type_input)
        #
        #
        #     temp_mask = np.load(mask_data_dir)
        #
        #     target_array = np.abs(np.squeeze(input)).sum(axis=0)
        #     measured_corrected_image = self._correct_bias_field_prediction(input=x_input, to_be_corrected=target_array)
        #     test_corrected_image = self._correct_bias_field_prediction(input=x0, to_be_corrected=target_array)
        #
        #     measured_corrected_image[np.isnan(measured_corrected_image)] = 0
        #     measured_corrected_image[np.isinf(measured_corrected_image)] = 0
        #
        #     test_corrected_image[np.isnan(test_corrected_image)] = 0
        #     test_corrected_image[np.isinf(test_corrected_image)] = 0
        #
        #     if self.debug:
        #         print('x0', x0.shape)
        #         print('A_tens', A_tensor.shape)
        #         print('meas', measured_corrected_image.shape)
        #         print('corimg', test_corrected_image.shape)
        #
        #     abs_measured_img = np.array(np.abs(measured_corrected_image))
        #     abs_measured_img = harray.scale_minpercentile(abs_measured_img, q=98)
        #     abs_test_img = np.array(np.abs(test_corrected_image))
        #     abs_test_img = harray.scale_minpercentile(abs_test_img, q=98)
        #
        #     fig_handle = hplotc.ListPlot([abs_measured_img * temp_mask, abs_test_img], vmin=(0, 1), cbar=True)
        #     fig_handle.figure.savefig(output_path)


class ExecutorInhomogRemovalGAN(ExecutorGAN):
    def __init__(self, model_path=None, config_file=None, **kwargs):
        super().__init__(model_path=model_path, config_file=config_file, **kwargs)
        self.transform_obj = None  # No idea what to do with this for now...

    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "inhomog":
            data_gen_sel = data_gen_inhomog.DataGeneratorInhomogRemovalH5
        elif dg_choice == "inhomogh5":
            data_gen_sel = data_gen_inhomog.DataGeneratorInhomogRemovalH5
        else:
            def data_gen_sel(x, **kwargs):
                return x

            print('EXEC AC GAN \t - No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorInhomogRemovalCycleGAN(ExecutorCycleGAN):
    def __init__(self, model_path=None, config_file=None, **kwargs):
        super().__init__(model_path=model_path, config_file=config_file, **kwargs)
        self.transform_obj = None  # No idea what to do with this for now...

    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "inhomogh5":
            data_gen_sel = data_gen_inhomog.DataGeneratorInhomogRemovalH5
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator  # ppff simple fucntion

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj
