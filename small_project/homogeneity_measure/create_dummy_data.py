import torch

import helper.array_transf
import helper.misc as hmisc

import helper.plot_class as hplotc
import tooling.shimming.b1shimming_single as mb1_single
import helper.array_transf as harray
import h5py
import numpy as np
import scipy.io
import skimage.transform
import skimage.metrics
import scipy.integrate
import small_project.sinp3.signal_equation as signal_eq

"""
Create a class that returns a varying amount of inhomogeneity for a given Rho and B1+ distr.
"""


class DummyVaryingSignalData:
    TR_se = 2500
    TE_se = 90
    T1_fat = 583
    T2_fat = 46
    T1_muscle = 1552
    T2_muscle = 23
    T1 = (T1_fat + T1_muscle) / 2
    T2 = (T2_fat + T2_muscle) / 2

    def __init__(self, rho, b1p, b1m, mask, min_degree=60, max_degree=90):
        # Rho and B1p need to have the same shape...
        self.rho = rho
        self.b1p = b1p
        self.b1m = b1m
        self.mask = mask
        # Center mask is used for shimming and scaling
        self.center_mask = self.create_center_mask(self.rho)
        self.b1p_shimmed = self.create_shimmed_b1p(self.b1p)

        # Settings for the variation in signal intensity
        # Degree = flip angle
        min_degree = min_degree
        max_degree = max_degree
        # Number of degrees between min and max
        n_degree = 30
        self.selected_flip_angles_degree = np.linspace(min_degree, max_degree, n_degree)

    @staticmethod
    def create_center_mask(x):
        # "SHIM" B1 mask
        # Doesnt reallllyy matter which image us used for this
        # We mainly need the shape
        n_y, n_x = x.shape[-2:]
        y_offset = np.random.randint(-n_y // 8, n_y // 8)
        x_offset = np.random.randint(-n_x // 8, n_x // 8)
        y_center = n_y // 2 + y_offset
        x_center = n_x // 2 + x_offset
        center_mask = np.zeros((n_y, n_x))
        delta_x = int(0.1 * n_x)
        delta_y = int(0.1 * n_y)
        center_mask[y_center - delta_y:y_center + delta_y, x_center - delta_x:x_center + delta_x] = 1
        return center_mask

    def create_varying_signal_maps(self, b1p_shimmed=None, rho=None, b1m=None, mask=None):
        # Set default values to class attributes
        if b1p_shimmed is None:
            b1p_shimmed = self.b1p_shimmed
        if rho is None:
            rho = self.rho
        if b1m is None:
            b1m = self.b1m
        if mask is None:
            mask = self.mask

        varying_signal_image = []
        varying_bias_field = []
        selected_flip_angles_rad = [np.deg2rad(x) for x in self.selected_flip_angles_degree]
        for fa_value in selected_flip_angles_rad:
            fa_map = self.get_flip_angle_map(b1p_shimmed, mask=self.center_mask, flip_angle=fa_value)

            t2_signal = signal_eq.get_t2_signal_general(fa_map, T1=self.T1, TR=self.TR_se,
                                                   TE=self.TE_se, beta=fa_map * 2, N=1, T2=self.T2)
            bias_field = np.abs(t2_signal * b1m).sum(axis=0)
            signal_array = np.abs(t2_signal * b1m).sum(axis=0) * rho * mask
            signal_array = harray.scale_minmax(signal_array)
            varying_signal_image.append(signal_array)
            varying_bias_field.append(bias_field)

        varying_signal_image = np.array(varying_signal_image)
        varying_bias_field = np.array(varying_bias_field)
        return varying_signal_image, varying_bias_field

    def create_shimmed_b1p(self, b1p_unshimmed=None):
        if b1p_unshimmed is None:
            b1p_unshimmed = self.b1p

        shimming_obj = mb1_single.ShimmingProcedure(b1p_unshimmed, self.center_mask,
                                                    relative_phase=True, str_objective='flip_angle')

        x_opt, final_value = shimming_obj.find_optimum()
        b1p_shimmed = harray.apply_shim(b1p_unshimmed, cpx_shim=x_opt)
        return b1p_shimmed

    @staticmethod
    def get_flip_angle_map(x, mask, flip_angle=np.pi/2):
        x_sub = x * mask
        # Mean over the masked area...
        x_mean = np.abs(x_sub.sum()) / np.sum(mask)
        flip_angle_map = np.abs(x) / x_mean * flip_angle
        return flip_angle_map




def apply_model(x, mask, modelrun_obj):
    with torch.no_grad():
        res = modelrun_obj.model_obj(x)

    if modelrun_obj.config_param['data']['target_type'] == 'rho':
        corrected_input = res.numpy()[0][0] * mask
        corrected_input = harray.scale_minmax(corrected_input)
        # Tried to do this again.. but then we get those super high values again...
        pred_biasfield = x / corrected_input * mask
        pred_biasfield = helper.array_transf.correct_inf_nan(pred_biasfield)
        pred_biasfield = harray.smooth_image(pred_biasfield, n_kernel=4)
        corrected_input = corrected_input / pred_biasfield
    else:
        pred_biasfield = res.numpy()[0][0]
        corrected_input = x / pred_biasfield * mask
        corrected_input = corrected_input.numpy()[0][0]
        corrected_input = helper.array_transf.correct_inf_nan(corrected_input)

    # Correct the output...
    hist_pred, hist_values = np.histogram(corrected_input[mask == 1].ravel(), bins=128, density=True, range=(0, 20))
    min_value = min(hist_values[1:][(hist_values[1:] > 0) & (hist_pred < 0.1)])
    corrected_input[corrected_input > min_value] = 1
    corrected_input[corrected_input < 0] = 0
    corrected_input = harray.scale_minmax(corrected_input)

    return corrected_input, pred_biasfield

