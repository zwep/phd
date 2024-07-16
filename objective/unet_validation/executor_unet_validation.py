# encoding: utf-8

import numpy as np
import os
import helper.plot_fun as hplotf
import helper.array_transf as harray
import torch

import model.UNet3D
import model.CycleGAN
import json

from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_regular import ExecutorRegular
import model.UNet as model_unet
import data_generator.UnetValidation as data_gen_unet_val
import inspect
import kornia.augmentation as kaugm
import imgaug.augmenters as iaa
import torch.autograd
import helper_torch.misc as htmisc


class ExecutorUnetValidation(ExecutorRegular):
    def __init__(self, model_path=None, config_file=None, **kwargs):
        super().__init__(model_path=model_path, config_file=config_file, **kwargs)
        self.model_obj = self.get_model()

        if self.model_obj:
            self.optimizer_obj = self.get_optimizer(self.model_obj.parameters())
        else:
            print('No model object has been defined, thus no optmizer object can be defined.')

    def get_model(self):
        model_choice = self.config_param['model']['model_choice']

        if model_choice.lower() == 'unet3d':
            model_obj = model.UNet3D.Unet3D(debug=self.debug, **self.config_param['model']['config_unet3d'])
            model_obj = model_obj.float()
            model_obj.to(self.device)
        else:
            model_obj = None

        if self.debug:
            print('INFO - REC - RX2TX - putting model to device ', self.device)

        return model_obj

    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data'].get('generator_choice', None)

        data_gen_sel = data_gen_unet_val.UnetValidation
        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    transform=self.transform_obj,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj

    def save_model(self, plot_name='final'):
        """

        :param plot_name: string appendix of plot figure
        :return:
        """
        # super().save_model()
        dir_weights = os.path.join(self.config_param['dir']['doutput'], self.name_model_weights)
        dir_temp_weights = os.path.join(self.config_param['dir']['doutput'], self.name_temp_weights)
        dir_history = os.path.join(self.config_param['dir']['doutput'], self.name_model_hist)

        # Save weights
        torch.save(self.model_obj.state_dict(), dir_weights)
        torch.save(self.temp_weights, dir_temp_weights)

        # Save history object
        serialized_json_history_obj = json.dumps(self.history_dict)
        with open(dir_history, 'w') as f:
            f.write(serialized_json_history_obj)

        doutput = self.config_param['dir']['doutput']

        x0, y0, y_pred0, _ = self.get_image_prediction()
        x0_time_ax = x0.shape[-1]
        y0_time_ax = y0.shape[-1]
        y_pred0_time_ax = y_pred0.shape[-1]
        print('Unet Validation - save model, data shapes', x0.shape, y0.shape, y_pred0.shape)
        plot_x = np.moveaxis(x0[0, :, :, x0_time_ax // 2 - 10:x0_time_ax // 2 + 10], -1, 0)[np.newaxis]
        plot_y = np.moveaxis(y0[0, :, :, y0_time_ax // 2 - 10:y0_time_ax // 2 + 10], -1, 0)[np.newaxis]
        plot_y_pred = np.moveaxis(y_pred0[0, :, :, y_pred0_time_ax // 2 - 10:y_pred0_time_ax // 2 + 10], -1, 0)[np.newaxis]
        print('Unet Validation - save model, plot shapes', plot_x.shape, plot_y.shape, plot_y_pred.shape)
        fig_handle_a = hplotf.plot_3d_list(plot_x, title='one input over multiple time frames', ax_off=True)
        fig_handle_a.savefig(os.path.join(doutput, f'input_{plot_name}.jpg'))
        fig_handle_b = hplotf.plot_3d_list(plot_y, title='one target over multiple time frames', ax_off=True)
        fig_handle_b.savefig(os.path.join(doutput, f'target_{plot_name}.jpg'))
        fig_handle_c = hplotf.plot_3d_list(plot_y_pred, title='one prediction over multiple time frames', ax_off=True)
        fig_handle_c.savefig(os.path.join(doutput, f'pred_{plot_name}.jpg'))
