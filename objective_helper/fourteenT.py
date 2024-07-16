import re
import os
import numpy as np
import helper.misc as hmisc
import helper.plot_class as hplotc
import h5py
import matplotlib.pyplot as plt
import scipy.io

# Misc code..
import scipy.optimize
import helper.array_transf as harray
import helper.plot_fun as hplotf
from objective_configuration.fourteenT import DDATA, DMASK_THOMAS, \
                                        ARRAY_SHAPE, X_METRIC, Y_METRIC, DDATA_POWER_DEPOS, \
    SUBDIR_OPTIM_SHIM, SUBDIR_RANDOM_SHIM, \
    MID_SLICE_OFFSET, COIL_NAME_ORDER_TRANSLATOR, TARGET_FLIP_ANGLE, RF_SCALING_FACTOR, \
    OPTIMAL_SHIM_SAR, OPTIMAL_SHIM_POWER
import helper.metric as hmetric
from objective_configuration.fourteenT import COIL_NAME_ORDER, COLOR_DICT, COLOR_MAP, TARGET_B1, TARGET_FLIP_ANGLE, WEIRD_RF_FACTOR
import pandas as pd


def flush_right_legend(legend_obj):
    plt.pause(0.1)
    max_shift = max([t.get_window_extent().width for t in legend_obj.get_texts()])
    for t in legend_obj.get_texts():
        t.set_ha('right')  # ha is alias for horizontalalignment
        temp_shift = max_shift - t.get_window_extent().width
        t.set_position((temp_shift, 0))


def convert_thomas_array_to_me(x):
    # Order should be (nz, ny, nx)
    # Magic conversion... this value is chosen as the index at which the brain stops at the top (feat-head dir)
    delta_top = ARRAY_SHAPE[0] - 71
    delta_bottom = ARRAY_SHAPE[0] - delta_top - x.shape[0]
    delta_shape = np.array(ARRAY_SHAPE) - np.array(x.shape)

    delta_side_1 = int(np.ceil(delta_shape[1] / 2))
    delta_side_2 = int(np.floor(delta_shape[1] / 2))

    new_x = np.pad(x, ((delta_bottom, delta_top),
                       (delta_side_1, delta_side_2),
                       (delta_side_1, delta_side_2)), mode='constant')
    return new_x


def _get_optimal_shim(result_dict, coil_name):
    # TODO dit is echt het meest irritante stukhje code
    # By    hand    from opt_shim_00
    opt_shim_selector = {'8 Channel Dipole Array 7T': 6,
                         '8 Channel Dipole Array': 4,
                         '16 Channel Loop Dipole Array': 4,
                         '15 Channel Dipole Array': 1,
                         '16 Channel Loop Array small': 4,
                         '8 Channel Loop Array big': 3}

    index_max = opt_shim_selector[coil_name]
    optimal_shim = result_dict['opt_shim'][index_max]
    return optimal_shim, index_max


class ReadMatData:
    h5_key = 'ProcessedData'

    def __init__(self, ddata, mat_file):
        # ddata is the folder where the mat_file is located
        # mat_file is the name of the coil plus an appendix
        # This object then allows for the reading of the .mat file in an easier way
        self.coil_name = mat_file.split('_')[0]
        self.coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[self.coil_name]
        self.mat_path = os.path.join(ddata, mat_file)
        # for sel_mat_file in mat_files:
        self.n_ports = int(re.findall('([0-9]+) Channel', self.coil_name)[0])
        # Global parameters... dit komt van de vertaling van de mat-files
        self.head_weight = 5  # kg
        if '7T' in self.coil_name:
            self.sigma_value = 0.413
        else:
            self.sigma_value = 0.5015  # target_mask is selected based on conductivity, gray matter

    def print_content_mat_obj(self):
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            hmisc.print_dict(mat_obj)

    def read_mat_object(self):
        # Read and load specific data points
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            sigma_array = np.array(mat_obj['sigma'])
            # select B1+, go to muT: It is given in Tesla, by multiplying it with 1e6 we have it in muT
            b1p_array = np.array(mat_obj['B1'][:, 0]['real'] + 1j * mat_obj['B1'][:, 0]['imag']) * 1e6
            b1m_array = np.array(mat_obj['B1'][:, 1]['real'] + 1j * mat_obj['B1'][:, 1]['imag']) * 1e6
            VOPm = np.array(mat_obj['VOPm']['real'] + 1j * mat_obj['VOPm']['imag'])
            Q10g = np.array(mat_obj['Q_10g_real']) + 1j * np.array(mat_obj['Q_10g_imag'])
            E_x = np.array(mat_obj['Ex']['real'] + 1j * mat_obj['Ex']['imag'])
            E_y = np.array(mat_obj['Ey']['real'] + 1j * mat_obj['Ey']['imag'])
            E_z = np.array(mat_obj['Ez']['real'] + 1j * mat_obj['Ez']['imag'])
            # Substrate mask..?
            substrate_mask = np.array(mat_obj['Substratemask'])
            target_mask = np.array(mat_obj['Targetmask'])

        sigma_mask = np.abs(sigma_array - self.sigma_value) < 0.001
        content_dict = {'sigma': sigma_array, 'Q10g': Q10g, 'b1p': b1p_array, 'b1m': b1m_array, 'VOP': VOPm,
                        'E_x': E_x, 'E_y': E_y, 'E_z': E_z}

        # Target mask - contains also the rungs/ring
        # Substrate mask - only the ring
        mask_dict = {'sigma_mask': sigma_mask, 'substrate_mask': substrate_mask,
                     'target_mask': target_mask}
        return content_dict, mask_dict

    def read_B1_object(self):
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            sigma_array = np.array(mat_obj['sigma'])
            # select B1+, go to muT: It is given in Tesla, by multiplying it with 1e6 we have it in muT
            b1p_array = np.array(mat_obj['B1'][:, 0]['real'] + 1j * mat_obj['B1'][:, 0]['imag']) * 1e6
            b1m_array = np.array(mat_obj['B1'][:, 1]['real'] + 1j * mat_obj['B1'][:, 1]['imag']) * 1e6
        content_dict = {'b1p': b1p_array, 'b1m': b1m_array, 'sigma': sigma_array}
        return content_dict

    def read_VOP_object(self):
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            VOPm = np.array(mat_obj['VOPm']['real'] + 1j * mat_obj['VOPm']['imag'])
        return {'VOP': VOPm}

    def read_Q_object(self):
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            Q10g = np.array(mat_obj['Q_10g_real']) + 1j * np.array(mat_obj['Q_10g_imag'])
        return {'Q10g': Q10g}

    def read_shim_object(self):
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            sigma_array = np.array(mat_obj['sigma'])
            # select B1+, go to muT: It is given in Tesla, by multiplying it with 1e6 we have it in muT
            b1p_array = np.array(mat_obj['B1'][:, 0]['real'] + 1j * mat_obj['B1'][:, 0]['imag']) * 1e6
            b1m_array = np.array(mat_obj['B1'][:, 1]['real'] + 1j * mat_obj['B1'][:, 1]['imag']) * 1e6
            Q10g = np.array(mat_obj['Q_10g_real']) + 1j * np.array(mat_obj['Q_10g_imag'])
            VOPm = np.array(mat_obj['VOPm']['real'] + 1j * mat_obj['VOPm']['imag'])
        content_dict = {'b1p': b1p_array, 'b1m': b1m_array, 'Q10g': Q10g, 'VOP': VOPm}
        return content_dict

    def read_subset_mat_object(self):
        # Read and load specific data points
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            sigma_array = np.array(mat_obj['sigma'])
            # select B1+, go to muT: It is given in Tesla, by multiplying it with 1e6 we have it in muT
            b1p_array = np.array(mat_obj['B1'][:, 0]['real'] + 1j * mat_obj['B1'][:, 0]['imag']) * 1e6
            b1m_array = np.array(mat_obj['B1'][:, 1]['real'] + 1j * mat_obj['B1'][:, 1]['imag']) * 1e6
            VOPm = np.array(mat_obj['VOPm']['real'] + 1j * mat_obj['VOPm']['imag'])
            E_x = np.array(mat_obj['Ex']['real'] + 1j * mat_obj['Ex']['imag'])
            E_y = np.array(mat_obj['Ey']['real'] + 1j * mat_obj['Ey']['imag'])
            E_z = np.array(mat_obj['Ez']['real'] + 1j * mat_obj['Ez']['imag'])
            substrate_mask = np.array(mat_obj['Substratemask'])
            target_mask = np.array(mat_obj['Targetmask'])

        sigma_mask = np.abs(sigma_array - self.sigma_value) < 0.001
        content_dict = {'sigma': sigma_array, 'b1p': b1p_array, 'b1m': b1m_array, 'VOP': VOPm,
                        'E_x': E_x, 'E_y': E_y, 'E_z': E_z}

        # Target mask - contains also the rungs/ring
        # Substrate mask - only the ring
        mask_dict = {'sigma_mask': sigma_mask, 'substrate_mask': substrate_mask,
                     'target_mask': target_mask}
        return content_dict, mask_dict

    def read_mask_object(self):
        # Read and load specific data points
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]
            sigma_array = np.array(mat_obj['sigma'])
            substrate_mask = np.array(mat_obj['Substratemask'])
            target_mask = np.array(mat_obj['Targetmask'])

        sigma_mask = np.abs(sigma_array - self.sigma_value) < 0.001

        # Target mask - contains also the rungs/ring
        # Substrate mask - only the ring
        mask_dict = {'sigma_mask': sigma_mask, 'substrate_mask': substrate_mask,
                     'target_mask': target_mask}
        return mask_dict

    def read_parameters(self):
        # We want to have a separate reader for all the (small) parameters in the matlab object
        with h5py.File(self.mat_path, 'r') as h5_obj:
            mat_obj = h5_obj[self.h5_key]

            resolution = float(mat_obj['res'][0][0])
            real_order = np.array(mat_obj['real_order']).ravel()
            VOP_overestimation = float(mat_obj['vops_overest_per'][0][0])
            VOP_method = np.array(mat_obj['vops_method']).ravel()
            VOP_calc = float(mat_obj['calc_VOPs'][0][0])
            SAR_avg = float(mat_obj['averageSAR'][0][0])
            SAR_mass = float(mat_obj['aveMass'][0][0])
            SAR_C = float(mat_obj['SAR_C'][0][0])
            substrate_sigma = float(mat_obj['substrate_sig_value'][0][0])
            median_filter = float(mat_obj['median_filter'][0][0])
            mask_substrate = float(mat_obj['mask_substrate'][0][0])

        content_dict = {'resolution': resolution, 'real_order': real_order, 'VOP_overestimation': VOP_overestimation,
                        'VOP_method': VOP_method, 'VOP_calc': VOP_calc, 'SAR_avg': SAR_avg, 'SAR_mass': SAR_mass,
                        'SAR_C': SAR_C, 'substrate_sigma': substrate_sigma, 'median_filter': median_filter,
                        'mask_substrate': mask_substrate}
        return content_dict

    def get_power_deposition_matrix(self, container, resolution):
        # Container should be the result of self.read_mat_object

        sigma_array = np.array(container['sigma'])
        #mid_slice = int(sigma_array.shape[0] / 2 + MID_SLICE_OFFSET[0])
        mid_slice = None
        sigma_array = sigma_array[mid_slice]
        Ex_flat = container['E_x'][:, mid_slice].reshape((self.n_ports, -1))
        Ey_flat = container['E_y'][:, mid_slice].reshape((self.n_ports, -1))
        Ez_flat = container['E_z'][:, mid_slice].reshape((self.n_ports, -1))

        sigma_array_flat = sigma_array.ravel()
        power_deposition_matrix = np.zeros((self.n_ports, self.n_ports), dtype=complex)
        for i in range(self.n_ports):
            for j in range(i, self.n_ports):
                print(f"Calculating ... ({i} / {j}) : ({self.n_ports} / {self.n_ports})", end='\r')
                E_xx = Ex_flat[i] * Ex_flat[j].conjugate()
                E_yy = Ey_flat[i] * Ey_flat[j].conjugate()
                E_zz = Ez_flat[i] * Ez_flat[j].conjugate()
                power_deposition_ij = sum(0.5 * sigma_array_flat * (E_xx + E_yy + E_zz)) * (resolution ** 3)
                power_deposition_matrix[i, j] = power_deposition_ij

        # Add the conjugate transpose to it...
        power_deposition_matrix = power_deposition_matrix + power_deposition_matrix.conjugate().T
        # Now normalize the diagonal.. because we counted it double in the above expression
        power_deposition_matrix[range(self.n_ports), range(self.n_ports)] = 0.5 * power_deposition_matrix[
            range(self.n_ports), range(self.n_ports)]
        return power_deposition_matrix

    def get_sigma_mask(self, sigma_array):
        sigma_shape = sigma_array.shape
        # Create a mask....
        sigma_mask = np.zeros(sigma_shape)
        sigma_index = np.abs(sigma_array - self.sigma_value) < 0.001
        sigma_mask[sigma_index] = 1
        return sigma_mask

    def get_tresholded_mask(self, sigma_array):
        n_slice, _, _ = sigma_array.shape
        sel_slice = n_slice // 2
        shim_mask = harray.get_treshold_label_mask(sigma_array[sel_slice], treshold_value=0.01 * sigma_array[sel_slice].mean())
        return shim_mask, sel_slice


class PathData:
    # Contains some names and such for plots and arrays...
    # These are dependent on the coil name.. and thus a class is more appropriate I think
    def __init__(self, ddest, coil_name, *args, **kwargs):
        """
        This class is used for the RF shims...

        :param ddest: destination directory
        :param coil_name: the coil name ...
        :param opt_shim_str: the file name of the optimized josn file
        :param args:
        :param kwargs:
        """
        if not os.path.isdir(os.path.join(ddest, coil_name)):
            os.makedirs(os.path.join(ddest, coil_name))

        # Below are all used to store numpy arrays. If stored without extension.. it gets an extention
        # When we want to load the image (png) version of it. We can easily append the extension
        self.path_snr_file = os.path.join(ddest, coil_name, f'SNR')
        self.path_sigma_file = os.path.join(ddest, coil_name, f'conductivity')
        self.path_b1_file = os.path.join(ddest, coil_name, 'opt_b1')
        self.path_sar_file = os.path.join(ddest, coil_name, 'opt_sar')
        self.path_opt_lambda_fig = os.path.join(ddest, coil_name, f'opt_lambda.png')
        #
        if 'rmse_power' in ddest:
            self.optimal_index_dict = OPTIMAL_SHIM_POWER
        elif 'rmse_sar' in ddest:
            self.optimal_index_dict = OPTIMAL_SHIM_SAR
        else:
            self.optimal_index_dict = {}

        optimal_file_index, optimal_lambda_index = self.optimal_index_dict[coil_name]
        optimal_file_index = str(optimal_file_index).zfill(2)
        json_path = os.path.join(ddest, SUBDIR_OPTIM_SHIM, coil_name, f'opt_shim_{optimal_file_index}.json')
        self.result_dict = hmisc.load_json(json_path)
        self.result_dict['opt_shim'] = [np.array(x[0]) + 1j * np.array(x[1]) for x in self.result_dict['opt_shim']]
        self.optimal_shim = self.result_dict['opt_shim'][optimal_lambda_index]
        print('We multiply the RF_SCALING_FACTOR with the optimal shim.')
        self.optimal_shim_rf_factor = RF_SCALING_FACTOR * self.optimal_shim


class DataCollector:
    def __init__(self, mat_reader, full_mask=False, type_mask='sigma', *args, **kwargs):
        """
        This class uses a Mat Reader object to collect all the necessary data in one object

        For most methods, given a specific shim setting, we can easily retrieve all shims
        :param mat_reader:
        :param full_mask:
        :param type_mask:
        :param args:
        :param kwargs:
        """
        self.mat_reader = mat_reader
        # Read data content...
        self.param_container = mat_reader.read_parameters()
        self.b1_container = mat_reader.read_B1_object()
        self.vop_container = mat_reader.read_VOP_object()
        self.mask_container = mat_reader.read_mask_object()

        self.full_mask = full_mask  # Selects the mid slice or the full brain
        self.type_mask = type_mask  # Select either sigma or brain mask..
        # Mask stuff...
        self.sigma_mask = self.mask_container['sigma_mask']
        self.brain_mask = self.mask_container['target_mask'] - self.mask_container['substrate_mask']
        self.selected_mask = self.get_mask()
        self.n_slice = self.selected_mask.shape[0]
        self.power_deposition_matrix = np.load(os.path.join(DDATA_POWER_DEPOS, self.mat_reader.coil_name + '.npy'))

    @staticmethod
    def select_array(x, mask_array):
        # This is going to be very ugly
        if x.ndim == 4:
            sel_x = x[:, mask_array == 1]
        elif x.ndim == 3:
            sel_x = x[mask_array == 1]
        else:
            sel_x = x
        return sel_x

    def get_mask(self):
        # Here we choose which mask we want/need..
        # This means that the mask we use will always be 3d
        # And that we dont use the sel-slice variable anymore
        if self.type_mask == 'sigma':
            chosen_mask = self.sigma_mask
        elif self.type_mask == 'brain':
            chosen_mask = self.brain_mask
        elif self.type_mask == 'thomas_mask':
            # Now... we need a kT points mask..
            chosen_mask = hmisc.load_array(DMASK_THOMAS)
        else:
            chosen_mask = np.ones(self.brain_mask.shape)

        # If we want a single slice for optimization and calculations..
        if self.full_mask:
            selected_mask = chosen_mask
        else:
            n_slice = chosen_mask.shape[0]
            sel_slice = n_slice // 2
            selected_mask = np.zeros(chosen_mask.shape)
            selected_mask[sel_slice] = chosen_mask[sel_slice]
        return selected_mask

    def get_shimmed_b1p(self, x_shim):
        correction_head_sar, optm_power_deposition = self.get_shim_power_deposition(x_shim)
        b1p_shimmed = (self.b1_container['b1p'].T @ x_shim).T

        return np.abs(b1p_shimmed), correction_head_sar, optm_power_deposition

    def get_shim_power_deposition(self, x_shim):
        optm_power_deposition = x_shim.T.conjugate() @ self.power_deposition_matrix @ x_shim
        optm_power_deposition = optm_power_deposition.real
        # % correction factor to set head SAR to 3.2 W/kg
        # [kg] [W / kg] = [W]
        # [W] /
        correction_head_sar = (self.mat_reader.head_weight * 3.2) / optm_power_deposition
        return correction_head_sar, optm_power_deposition

    def get_peak_SAR_VOP(self, x_shim):
        vop_shimmed = np.einsum("d, dcn, c -> n", x_shim.conjugate(), self.vop_container['VOP'], x_shim)
        correction_head_sar, optm_power_deposition = self.get_shim_power_deposition(x_shim)
        peakSAR = np.max(vop_shimmed.real)
        peakSAR_normalized = (np.max(vop_shimmed.real) * correction_head_sar).real
        return peakSAR, peakSAR_normalized, optm_power_deposition

    def get_peak_SAR_Q(self, x_shim):
        Q_container = self.mat_reader.read_Q_object()
        self.Q_array = Q_container['Q10g']
        vop_shimmed = np.einsum("d, dcxyz, c -> xyz", x_shim.conjugate(), self.Q_array, x_shim)
        correction_head_sar, optm_power_deposition = self.get_shim_power_deposition(x_shim)
        peakSAR = np.max(vop_shimmed.real)
        peakSAR_normalized = (np.max(vop_shimmed.real) * correction_head_sar).real
        return peakSAR, peakSAR_normalized, optm_power_deposition

    def get_cov_b1p(self, x_shim):
        b1p_shimmed, correction_head_sar, optm_power_deposition = self.get_shimmed_b1p(x_shim)
        sel_b1p = self.select_array(b1p_shimmed, mask_array=self.selected_mask)

        cov_b1p = hmetric.coefficient_of_variation(sel_b1p) * 100
        cov_b1p_normalized = hmetric.coefficient_of_variation(sel_b1p * correction_head_sar ** 2) * 100
        return cov_b1p, cov_b1p_normalized

    def get_nrmse_b1p(self, x_shim, x_target=1):
        b1p_shimmed, correction_head_sar, optm_power_deposition = self.get_shimmed_b1p(x_shim)
        sel_b1p = self.select_array(b1p_shimmed, mask_array=self.selected_mask)
        rmse, nrmse = hmetric.normalized_rmse(sel_b1p, x_target)
        avg_b1 = np.mean(np.abs(sel_b1p))

        rmse_normalized, nrmse_normalized = hmetric.normalized_rmse(sel_b1p * correction_head_sar ** 2, x_target)
        avg_b1_normalized = np.mean(np.abs(sel_b1p * correction_head_sar ** 2))
        return rmse, nrmse, avg_b1, rmse_normalized, nrmse_normalized, avg_b1_normalized


class OptimizeData(DataCollector):
    # This object used the mat_reader to immediately read a set of matlab objects
    def __init__(self, ddest, mat_reader, full_mask=False, type_mask='thomas_mask', objective_str='rmse_power'):
        super().__init__(mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)

        # Get the system matrix for this problem...
        self.system_matrix = self.select_array(self.b1_container['b1p'], self.selected_mask).T
        self.lambda_range = np.linspace(0, 1/10., 25)
        self.target_vector = np.ones(self.system_matrix.shape[0])
        self._lambda_value = 0
        # Select the correct objective function
        if objective_str == 'mae_power':
            self.objective_fun = self.objective_mae
        elif objective_str == 'rmse_power':
            self.objective_fun = self.objective_rmse
        elif objective_str == 'rmse_sar':
            self.objective_fun = self.objective_rmse_sar
        else:
            self.objective_fun = None
            print("Unknown objective function")

        # Create the location where we want to store the optimal json files
        self.ddest_optim_shim_coil = os.path.join(ddest, SUBDIR_OPTIM_SHIM, mat_reader.coil_name)
        self.ddest_random_shim_coil = os.path.join(ddest, SUBDIR_RANDOM_SHIM, mat_reader.coil_name)
        if not os.path.isdir(self.ddest_optim_shim_coil):
            os.makedirs(self.ddest_optim_shim_coil)

    def report_parameters(self):
        print(self.mat_reader.coil_name)
        print('VOP overestimation :', self.param_container['VOP_overestimation'])

    def get_result_container_Q(self, x_opt, scaling_factor=1, append_key_bool=False):
        x_opt = x_opt * scaling_factor
        peakSAR, peakSAR_normalized, optm_power_deposition = self.get_peak_SAR_Q(x_shim=x_opt)
        cov_b1p, cov_b1p_normalized = self.get_cov_b1p(x_shim=x_opt)
        rmse, nrmse, avg_b1, rmse_normalized, nrmse_normalized, avg_b1_normalized = self.get_nrmse_b1p(x_shim=x_opt, x_target=scaling_factor)
        # Norm of the drive vector
        norm_x_opt = np.sum(np.abs(x_opt) ** 2)
        correction_head_sar, optm_power_deposition = self.get_shim_power_deposition(x_opt)
        norm_x_opt_normalized = np.sum(np.abs(x_opt * correction_head_sar) ** 2)
        #
        container = {'b1p_cov': cov_b1p, 'peak_SAR': peakSAR, 'opt_shim': x_opt,
                     'norm_power': norm_x_opt, 'b1_avg': avg_b1,
                     'b1p_nrmse': nrmse, 'residual': rmse, 'head_SAR': optm_power_deposition / 5,  # divide by head weight
                     'power_deposition': optm_power_deposition}

        container_normalized = {'b1p_cov_normalized': cov_b1p_normalized, 'peak_SAR_normalized': peakSAR_normalized,
                                'norm_power_normalized': norm_x_opt_normalized,
                                'b1_avg_normalized': avg_b1_normalized, 'b1p_nrmse_normalized': nrmse_normalized, 'residual_normalized': rmse_normalized}

        container.update(container_normalized)

        if append_key_bool:
            key_list = list(container.keys())
            for i_key in key_list:
                if not 'opt_shim' in i_key:
                    new_key_name = f'{i_key}_{TARGET_FLIP_ANGLE}_degrees'
                    container[new_key_name] = container.pop(i_key)

        return container

    def get_result_container(self, x_opt, scaling_factor=1, append_key_bool=False):
        x_opt = x_opt * scaling_factor
        peakSAR, peakSAR_normalized, optm_power_deposition = self.get_peak_SAR_VOP(x_shim=x_opt)
        cov_b1p, cov_b1p_normalized = self.get_cov_b1p(x_shim=x_opt)
        rmse, nrmse, avg_b1, rmse_normalized, nrmse_normalized, avg_b1_normalized = self.get_nrmse_b1p(x_shim=x_opt, x_target=scaling_factor)
        # Norm of the drive vector
        norm_x_opt = np.sum(np.abs(x_opt) ** 2)
        correction_head_sar, optm_power_deposition = self.get_shim_power_deposition(x_opt)
        norm_x_opt_normalized = np.sum(np.abs(x_opt * correction_head_sar) ** 2)
        #
        container = {'b1p_cov': cov_b1p, 'peak_SAR': peakSAR, 'opt_shim': x_opt,
                     'norm_power': norm_x_opt, 'b1_avg': avg_b1,
                     'b1p_nrmse': nrmse, 'residual': rmse, 'head_SAR': optm_power_deposition / 5,
                     'power_deposition': optm_power_deposition}

        container_normalized = {'b1p_cov_normalized': cov_b1p_normalized, 'peak_SAR_normalized': peakSAR_normalized,
                                'norm_power_normalized': norm_x_opt_normalized,
                                'b1_avg_normalized': avg_b1_normalized, 'b1p_nrmse_normalized': nrmse_normalized, 'residual_normalized': rmse_normalized}

        container.update(container_normalized)

        if append_key_bool:
            key_list = list(container.keys())
            for i_key in key_list:
                if not 'opt_shim' in i_key:
                    new_key_name = f'{i_key}_{TARGET_FLIP_ANGLE}_degrees'
                    container[new_key_name] = container.pop(i_key)

        return container

    def solve_trade_off_objective(self, max_iter=500):
        result = []
        for lambda_value in self.lambda_range:
            print(f'Lambda {lambda_value} / {self.lambda_range.max()}', end='\r')
            random_amp = np.random.uniform(low=0, high=5, size=self.mat_reader.n_ports)
            random_phase = np.random.uniform(low=0, high=2*np.pi, size=self.mat_reader.n_ports)
            x_init = np.concatenate([random_amp, random_phase])
            self._lambda_value = lambda_value
            x_opt = scipy.optimize.minimize(fun=self.objective_fun, x0=x_init, tol=1e-4, method='CG',
                                                options={"maxiter": max_iter})
            x_opt_cpx = self.convert_to_cpx(x_opt.x)
            result_dict = self.get_result_container(x_opt_cpx)
            result.append(result_dict)
            result_dict_scaled = self.get_result_container(x_opt_cpx, scaling_factor=RF_SCALING_FACTOR,
                                                           append_key_bool=True)
            result.append(result_dict_scaled)

        result_dict_list = hmisc.listdict2dictlist(result)
        return result_dict_list

    def get_random_shim_setting(self):
        random_amplitude = np.random.uniform(low=0, high=8, size=self.mat_reader.n_ports)
        random_phase = 2 * np.pi * np.random.uniform(size=self.mat_reader.n_ports)
        random_shim_setting = random_amplitude * np.exp(1j * random_phase)
        # random_shim_setting = random_shim_setting.reshape((-1, 1))
        return random_shim_setting

    def get_binned_random_shim(self, x):
        amp_factor = int(x / 6000)
        random_amplitude = np.random.uniform(low=amp_factor, high=amp_factor + 1, size=self.mat_reader.n_ports)
        random_phase = 2 * np.pi * np.random.uniform(low=0, high=1, size=self.mat_reader.n_ports)
        random_shim_setting = random_amplitude * np.exp(1j * random_phase)
        return random_shim_setting

    @staticmethod
    def convert_to_cpx(x):
        n = len(x)
        x_real = x[:n // 2]
        x_imag = x[n // 2:]
        # return x_real + 1j * x_imag
        return x_real * np.exp(1j * x_imag)

    def objective_mae(self, x):
        x_cpx = self.convert_to_cpx(x)
        system_shimmed = np.abs(self.system_matrix @ x_cpx)
        mae = np.mean(np.abs(system_shimmed - TARGET_B1)) + self._lambda_value * np.sum(np.abs(x_cpx) ** 2)
        return mae

    def objective_rmse(self, x):
        x_cpx = self.convert_to_cpx(x)
        system_shimmed = np.abs(self.system_matrix @ x_cpx)
        rmse = np.sqrt(np.mean(np.abs(system_shimmed - TARGET_B1)**2)) + self._lambda_value * np.sum(np.abs(x_cpx) ** 2)
        return rmse

    def objective_rmse_sar(self, x):
        x_cpx = self.convert_to_cpx(x)
        sar_penalty = np.einsum("d, dcn, c -> n", x_cpx.conjugate(), self.vop_container['VOP'], x_cpx)
        peakSAR = np.max(sar_penalty.real)
        system_shimmed = np.abs(self.system_matrix @ x_cpx)
        rmse = np.sqrt(np.mean(np.abs(system_shimmed - TARGET_B1) ** 2)) + self._lambda_value * peakSAR
        return rmse


class StoreOptimizeData(DataCollector, PathData):
    def __init__(self, ddest, mat_reader, full_mask=False, type_mask='sigma'):
        DataCollector.__init__(self, mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)
        PathData.__init__(self, ddest=ddest, coil_name=mat_reader.coil_name)

    def store_SNR(self):
        """Store SNR"""  # % Show SNR (unit uT/sqrt(W))

        b1m_array = self.b1_container['b1m']
        b1m_array_flat = b1m_array.reshape((self.mat_reader.n_ports, -1))
        inv_power_deposition_matrix = np.linalg.inv(self.power_deposition_matrix)

        # B1m cdot P^(-1) * B1m
        SNR_flat = np.einsum("nc,cd,dn->n", b1m_array_flat.T.conjugate(), inv_power_deposition_matrix, b1m_array_flat)
        # Take the square root...
        SNR_flat = np.abs(np.sqrt(SNR_flat))
        # Reshape to 3D thing..
        SNR = SNR_flat.reshape(b1m_array.shape[1:])
        np.save(self.path_snr_file, SNR)

    def store_conductivity(self):
        """Store conductivity"""
        # Display slices from sigma..
        sigma_array = self.b1_container['sigma']
        np.save(self.path_sigma_file, sigma_array)

    def store_optimal_b1(self, sel_shim=None):
        # Here we store the B1 normalized for two ways.
        # I
        if sel_shim is None:
            # sel_shim = self.optimal_shim
            # Changes it to the scaled version
            sel_shim = self.optimal_shim_rf_factor
        b1p_shimmed, correction_head_sar, optm_power_deposition = self.get_shimmed_b1p(sel_shim)
        np.save(self.path_b1_file + '_forward_power', b1p_shimmed)
        np.save(self.path_b1_file + '_head_sar', b1p_shimmed * optm_power_deposition ** 2)

    def get_shimmed_sar_q(self, x_shim):
        # Hopefully this way we will save some memory...
        Q_container = self.mat_reader.read_Q_object()
        correction_head_sar, optm_power_deposition = self.get_shim_power_deposition(x_shim)
        sar_shimmed = np.einsum("d, dczyx, c -> zyx", x_shim.conjugate(), Q_container['Q10g'], x_shim)
        return sar_shimmed, correction_head_sar, optm_power_deposition

    def store_optimal_sar(self, sel_shim=None):
        if sel_shim is None:
            # sel_shim = self.optimal_shim
            sel_shim = self.optimal_shim_rf_factor
        shimmed_sar, correction_head_sar, optm_power_deposition = self.get_shimmed_sar_q(sel_shim)
        np.save(self.path_sar_file + '_forward_power', shimmed_sar)
        np.save(self.path_sar_file + '_head_sar', shimmed_sar * correction_head_sar)


class VisualizeData(DataCollector, PathData):
    def __init__(self, ddest, mat_reader, full_mask=False, type_mask='sigma',
                 opt_subdir='optimal_shim_forward_power'):
        DataCollector.__init__(self, mat_reader=mat_reader, full_mask=full_mask, type_mask=type_mask)
        PathData.__init__(self, ddest=ddest, coil_name=mat_reader.coil_name,
                          opt_subdir=opt_subdir)

        self.thomas_mask_array = hmisc.load_array(DMASK_THOMAS)

    def plot_optimal_lambda(self):
        x_coord = self.result_dict[X_METRIC]
        y_coord = self.result_dict[Y_METRIC]
        curvature, index_max = hmisc.get_maximum_curvature(x_coord, y_coord)
        fig, ax = plt.subplots(2)
        ax[0].scatter(x_coord, y_coord)
        for ii in np.arange(0, len(x_coord), 5):
            ax[0].text(x_coord[ii], y_coord[ii] * 1.05, f'{ii}')
        ax[0].scatter(x_coord[index_max], y_coord[index_max], c='r')
        ax[0].set_title('B1p+ vs peak SAR')
        ax[1].plot(curvature, 'ro')
        ax[1].set_title('curvature')
        fig.savefig(self.path_opt_lambda_fig)
        return fig

    def plot_SNR(self):
        SNR = np.load(self.path_snr_file + '.npy')[::-1]
        SNR_plot_list = hplotf.get_all_mid_slices(SNR, offset=MID_SLICE_OFFSET)
        fig_obj = hplotc.ListPlot([SNR_plot_list], ax_off=True, cbar=True, wspace=0.5,
                                  title=self.mat_reader.coil_name, cmap=COLOR_MAP)
        fig_obj.figure.savefig(self.path_snr_file + '.png')
        return SNR_plot_list

    def plot_conductivity(self):
        """ Not really needed..."""
        sigma_array = np.load(self.path_sigma_file + '.npy')[::-1]
        sigma_plot_list = hplotf.get_all_mid_slices(sigma_array, offset=MID_SLICE_OFFSET)
        fig_obj = hplotc.ListPlot([sigma_plot_list], ax_off=True,
                                  title=self.mat_reader.coil_name)
        fig_obj.figure.savefig(self.path_sigma_file + '.png')
        return sigma_plot_list

    def plot_optimal_b1(self, str_normalization):
        self.b1 = np.load(self.path_b1_file + str_normalization + '.npy')
        b1_plot_list = hplotf.get_all_mid_slices(self.b1[::-1], offset=MID_SLICE_OFFSET)
        fig_obj = hplotc.ListPlot([b1_plot_list], ax_off=True, cbar=True, wspace=0.5,
                                  title=self.mat_reader.coil_name, cmap=COLOR_MAP)
        fig_obj.figure.savefig(self.path_b1_file + '.png')
        return b1_plot_list

    def plot_optimal_sar(self, str_normalization):
        self.sar = np.load(self.path_sar_file + str_normalization + '.npy')
        sar_plot_list = hplotf.get_all_mid_slices(self.sar[::-1], offset=MID_SLICE_OFFSET)
        fig_obj = hplotc.ListPlot([sar_plot_list], ax_off=True, cbar=True, wspace=0.5,
                                  augm='np.abs',
                                  title=self.mat_reader.coil_name, cmap=COLOR_MAP)
        fig_obj.figure.savefig(self.path_sar_file + '.png')
        return sar_plot_list


class VisualizeAllMetrics:
    def __init__(self, ddest, opt_shim_str='opt_shim_00'):
        """
        This class is used to visualize all the metrics INSIDE a specific optimal shim run.

        It is mainly used to demonstrate the trade off curves and SAR histograms

            data: contains the files where the mat-data is located. Which is mostly used to extract the relevant names
            djson: is the folder that contains the .json files with optimal shimming coefs
        """
        self.ddest = ddest
        self.opt_shim_str = opt_shim_str
        self.optimized_json_data = self.load_json_files(subdir=SUBDIR_OPTIM_SHIM, cpx_key=opt_shim_str)
        # We are not loading this object, becaues it can be quite large
        self.random_json_data = None

        # Select the optimal shim
        if 'rmse_power' in ddest:
            self.optimal_index_dict = OPTIMAL_SHIM_POWER
        elif 'rmse_sar' in ddest:
            self.optimal_index_dict = OPTIMAL_SHIM_SAR
        else:
            self.optimal_index_dict = {}

    @staticmethod
    def _get_freedman_bins(x):
        n = len(x)
        iqr_manual = np.quantile(x, q=[.25, .75])
        IQR = np.diff(iqr_manual)[0]
        # Using Friedman Diaconis rule...
        bin_width = 2 * IQR / n ** (1 / 3)
        bins = np.arange(min(x), max(x), bin_width)
        return bins

    @staticmethod
    def _load_json(json_path, cpx_key='opt_shim'):
        if 'opt' in cpx_key:
            cpx_key = 'opt_shim'
        elif 'random' in cpx_key:
            cpx_key = 'random_shim'
        else:
            print("Wrong cpx key chosen ", cpx_key)
        # Slight change to load_json...
        # Because we want to convert one component to complex data...
        result_dict = hmisc.load_json(json_path)
        result_dict[cpx_key] = [np.array(x[0]) + 1j * np.array(x[1]) for x in result_dict[cpx_key]]
        return result_dict

    def load_json_files(self, subdir='random_shim', cpx_key='random_shim'):
        result_all_files = {}
        for coil_name in COIL_NAME_ORDER:
            json_path = os.path.join(self.ddest, subdir, coil_name, f'{cpx_key}.json')
            if os.path.isfile(json_path):
                result_dict = self._load_json(json_path, cpx_key=cpx_key)
            else:
                print(f'No file found : {json_path}')
                result_dict = None
            result_all_files[coil_name] = result_dict

        return result_all_files

    def visualize_optimal_trade_off(self, x_metric=None, y_metric=None, fig=None, alpha=1, plot_text=False):
        if x_metric is None:
            x_metric = X_METRIC
        if y_metric is None:
            y_metric = Y_METRIC

        if fig is None:
            fig, ax = plt.subplots()
            plot_label = True
        else:
            ax = fig.axes[0]
            plot_label = False

        for k, v in self.optimized_json_data.items():
            color = COLOR_DICT[k]
            if plot_label is False:
                k = None
            ax.plot(v[x_metric], v[y_metric], 'o', label=k, color=color, alpha=alpha)
            if plot_text:
                ax.text(s='0', x=v[x_metric][0], y=v[y_metric][0])
                ax.text(s=str(len(v[x_metric])), x=v[x_metric][-1], y=v[y_metric][-1])
            ax.set_xlabel('NRMSE [%]')
            ax.set_ylabel('peak SAR (10g) [W/kg]')
            if plot_label:
                legend_obj = ax.legend()
                for lh in legend_obj.legendHandles:
                    lh.set_alpha(1)
        # ax.set_xlim(0, 200)
        # ax.set_ylim(0, 5)
        fig.savefig(os.path.join(self.ddest, f'L_curve_{self.opt_shim_str}.png'), bbox_inches='tight')
        return fig

    def visualize_hist_sar(self, fig=None, max_points=10000, key='peak_SAR'):
        if self.random_json_data is None:
            # Loading this conditionally since it can take some time..
            self.random_json_data = self.load_json_files(subdir=SUBDIR_RANDOM_SHIM, cpx_key='random_shim')

        if fig is None:
            fig, ax = plt.subplots(2, 3, figsize=(12, 8))
            ax = ax.ravel()
        else:
            ax = fig.get_axes()

        for ii, sel_coil in enumerate(COIL_NAME_ORDER):
            coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[sel_coil]
            sel_color = COLOR_DICT[sel_coil]
            if self.random_json_data[sel_coil] is not None:
                hist_data = self.random_json_data[sel_coil][key][:max_points]
                bins = self._get_freedman_bins(hist_data)
                ax[ii].hist(hist_data, alpha=0.5, bins=bins, color=sel_color, label=coil_plot_name)
                legend_obj = ax[ii].legend()
                flush_right_legend(legend_obj)
                #ax[ii].set_xlim(0, 50)
                #ax[ii].set_ylim(0, 750)
                ax[ii].set_xlabel('peak SAR')
                ax[ii].set_ylabel('Number of voxels')
        return fig

    def visualize_hist_sar_single_figure(self, max_points=10000):
        if self.random_json_data is None:
            # Loading this conditionally since it can take some time..
            self.random_json_data = self.load_json_files(subdir=SUBDIR_RANDOM_SHIM, cpx_key='random_shim')
        fig, ax = plt.subplots(figsize=(12, 8))

        for ii, sel_coil in enumerate(COIL_NAME_ORDER):
            coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[sel_coil]
            sel_color = COLOR_DICT[sel_coil]
            hist_data = self.random_json_data[sel_coil]['peak_SAR_normalized'][:max_points]
            bins = self._get_freedman_bins(hist_data)
            ax.hist(hist_data, label=coil_plot_name, bins=bins, alpha=0.25, color=sel_color)

        ax.set_xlabel('SAR_10g (W kg -1)')
        ax.set_ylabel('#')
        ax.legend()
        fig.savefig(os.path.join(self.ddest, f'peak_sar_histogram_single_figure.png'), bbox_inches='tight')
        return fig


class KtImage:
    def __init__(self, ddata, ddest, coil_name, weird_rf_factor=1, flip_angle_factor=1, load_Q=False):
        """
        This class should deal with the loading of all the simulated data by Thomas
        This is

        :param ddata:
        :param ddest:
        :param coil_name:
        :param weird_rf_factor:
        :param flip_angle_factor:
        """
        self.ddata = ddata
        self.base_dir_name = os.path.dirname(ddata)
        self.coil_name = coil_name
        self.coil_plot_name = COIL_NAME_ORDER_TRANSLATOR[self.coil_name]
        self.mat_file = coil_name + '_ProcessedData.mat'
        self.flip_angle_factor = flip_angle_factor  # Used to set a certain degree...
        self.weird_rf_factor = weird_rf_factor  # Used to amplify the solution... because code

        # Define all files..
        mat_files = os.listdir(ddata)
        self.debug_design_files = [x for x in mat_files if
                                   x.startswith('debug') and self._get_coil_name(x) == coil_name]
        self.debug_design_files = sorted(self.debug_design_files, key=self._file_sorter)
        self.output_design_files = [x for x in mat_files if
                                    x.startswith('output') and self._get_coil_name(x) == coil_name]
        self.output_design_files = sorted(self.output_design_files, key=self._file_sorter)

        # Create some output dirs..
        self.ddest = os.path.join(ddest, coil_name)
        if not os.path.isdir(self.ddest):
            os.makedirs(self.ddest)

        # These are numpy array files, no extension needed... those are added on later when loading
        self.path_sar_file = os.path.join(self.ddest, 'opt_sar')
        self.path_flip_angle_file = os.path.join(self.ddest, 'opt_flip_angle')

        # Load the Thomas mask
        self.thomas_mask_array = hmisc.load_array(DMASK_THOMAS)
        # Create a mat reader so that we can load the VOP data
        self.mat_reader = ReadMatData(ddata=DDATA, mat_file=self.mat_file)
        # Not loaded data or objects.....
        VOP_container = self.mat_reader.read_VOP_object()
        self.VOP_array = VOP_container['VOP']
        if load_Q:
            Q_container = self.mat_reader.read_Q_object()
            self.Q_array = Q_container['Q10g']
        self.power_deposition_matrix = np.load(os.path.join(DDATA_POWER_DEPOS, self.mat_reader.coil_name + '.npy'))

    def get_nrmse_peak_sar(self):
        # Gets all the NRMSE and peakSAR over alle the output design files...
        temp_peak_sar = []
        temp_nrmse = []
        for i_file in self.output_design_files:
            max_peak_sar, head_sar_correction, avg_power_deposition = self.get_peak_SAR(i_file)
            rmse, nrmse = self.get_flip_angle_nrmse(i_file)
            temp_peak_sar.append(max_peak_sar)
            temp_nrmse.append(nrmse)
        return temp_nrmse, temp_peak_sar

    def get_nrmse_peak_sar_Q(self):
        # Gets all the NRMSE and peakSAR over alle the output design files...
        temp_peak_sar = []
        temp_nrmse = []
        for i_file in self.output_design_files:
            max_peak_sar, head_sar_correction, avg_power_deposition = self.get_peak_SAR_Q(i_file)
            rmse, nrmse = self.get_flip_angle_nrmse(i_file)
            temp_peak_sar.append(max_peak_sar)
            temp_nrmse.append(nrmse)
        return temp_nrmse, temp_peak_sar

    def time_averaged_power_deposition_matrix(self, file_name):
        # This thing is something like (ncoil x ncoil)
        # Now we need to compute teh amount of power it deposits
        # when we are dealing with time varying rf coefficients.. should be hard
        rf_waveform = self.get_unique_pulse_settings(file_name)
        avg_optm_power_deposition = 0
        for x_shim in rf_waveform.T:
            avg_optm_power_deposition += (x_shim.conjugate() @ self.power_deposition_matrix @ x_shim)
        avg_optm_power_deposition /= len(rf_waveform)
        avg_optm_power_deposition = avg_optm_power_deposition.real
        #-----
        # % correction factor to set head SAR to 3.2 W/kg
        head_sar_correction = (self.mat_reader.head_weight * 3.2) / avg_optm_power_deposition
        return head_sar_correction, avg_optm_power_deposition

    @staticmethod
    def _get_kt_num(file_name):
        split_file_name = file_name.split('_')
        kt_str = hmisc.get_base_name(split_file_name[3])
        kt_num = int(re.findall('([0-9]*)Kt', kt_str)[0])
        return kt_num

    def _file_sorter(self, file_name):
        # if self.base_dir_name in ['results_kt_beta', 'results_1kt_beta_power']:
        if 'beta' in file_name:
            numeric_value = float(re.findall('_([0-9]*\.[0-9]*)beta', file_name)[0])
        # We dont use this one anymore...
        # elif self.base_dir_name in ['results_kt_beta_adjust']:
        #     numeric_value = int(re.findall('_([0-9]*)beta', file_name)[0])
        else:
            numeric_value = self._get_kt_num(file_name)
        return numeric_value

    @staticmethod
    def _get_coil_name(file_name):
        split_file_name = file_name.split('_')
        coil_name = split_file_name[2]
        return coil_name

    @staticmethod
    def get_peak_locations(waveform):
        pulse_locations = np.diff(np.abs(waveform))
        pos_change_locations = np.argwhere(pulse_locations > 0)
        # Reshape and add one due to the shape change in np.diff()
        pos_change_locations = pos_change_locations.ravel() + 1
        return pos_change_locations

    @staticmethod
    def _output2debug_file(file_name):
        return re.sub('output_', 'debug_', file_name)

    def _get_kt_file(self, kt_spoke):
        # kt_spoke is a number...
        kt_spoke_files = [x for x in self.output_design_files if self._get_kt_num(x) == kt_spoke]
        # Now the questions is... what the length of this is..
        if len(kt_spoke_files) > 1:
            print("Number of spoke files is larger than 1")
            print(kt_spoke_files)
            sel_spoke_file = None
        elif len(kt_spoke_files) == 0:
            print("There are no files..")
            sel_spoke_file = None
        else:
            print("We have one spoke file. Selecting it")
            sel_spoke_file = kt_spoke_files[0]
        return sel_spoke_file

    def get_pulse(self, file_name):
        sel_mat_file = os.path.join(self.ddata, file_name)
        mat_obj = scipy.io.loadmat(sel_mat_file)
        rf_waveform = mat_obj['output']['RF_Waveforms_mT'][0][0].T
        return rf_waveform

    def get_unique_pulse_settings(self, file_name):
        rf_waveform = self.get_pulse(file_name)
        # Removing this, because I want to "use" the whole pulse?
        peak_locations = self.get_peak_locations(rf_waveform[0])
        return self.weird_rf_factor * self.flip_angle_factor * rf_waveform[:, peak_locations]

    def read_peak_SAR(self, file_name):
        sel_mat_file = os.path.join(self.ddata, file_name)
        mat_obj = scipy.io.loadmat(sel_mat_file)
        return mat_obj['output'][0][0]['rf'][0][0]['peakSAR'][0][0]

    def get_peak_SAR(self, file_name):
        rf_waveform = self.get_unique_pulse_settings(file_name)
        # nc -> n pulses, c coils
        # cdz -> c coils, d coils, z number of VOPs
        # dn -> d coils, n pulses
        # Returns n pulses by z number of VOPs
        shimmed_VOP = np.einsum("nc,cdz,dn->nz", rf_waveform.T.conjugate(), self.VOP_array, rf_waveform)
        # print('Hi, I am taking the mean over the number of pulses. Then take the maximum of the real part')
        # print(' I think this is wrong, but I am not sure. Looks weird to average something like peak SAR..')
        max_peak_sar = np.max(shimmed_VOP.mean(axis=0).real)
        head_sar_correction, avg_power_deposition = self.time_averaged_power_deposition_matrix(file_name)
        # print(f'Measured peak SAR by Thomas {self.read_peak_SAR(file_name)} and by me {max_peak_sar}')
        return max_peak_sar, head_sar_correction, avg_power_deposition

    def get_peak_SAR_Q(self, file_name):
        rf_waveform = self.get_unique_pulse_settings(file_name)
        # nc -> n pulses, c coils
        # cdz -> c coils, d coils, z number of VOPs
        # dn -> d coils, n pulses
        # Returns n pulses by z number of VOPs
        shimmed_VOP = np.einsum("nc,cdxyz,dn->nxyz", rf_waveform.T.conjugate(), self.Q_array, rf_waveform)
        # print('Hi, I am taking the mean over the number of pulses. Then take the maximum of the real part')
        # print(' I think this is wrong, but I am not sure. Looks weird to average something like peak SAR..')
        # (n_s, n_V) -
        max_peak_sar = np.max(shimmed_VOP.mean(axis=0).real)
        head_sar_correction, avg_power_deposition = self.time_averaged_power_deposition_matrix(file_name)
        # print(f'Measured peak SAR by Thomas {self.read_peak_SAR(file_name)} and by me {max_peak_sar}')
        return max_peak_sar, head_sar_correction, avg_power_deposition

    def get_flip_angle_nrmse(self, file_name):
        kt_spoke_array = np.abs(self.get_flip_angle_map(file_name))
        # Using flip angle factor here because kt spokes array is optimized for 1 degree
        rmse, nrmse = hmetric.normalized_rmse(kt_spoke_array[self.thomas_mask_array == 1], self.flip_angle_factor)
        return rmse, nrmse

    def get_avg_norm_waveform(self, file_name):
        sel_mat_file = os.path.join(self.ddata, file_name)
        mat_obj = scipy.io.loadmat(sel_mat_file)
        rf_waveform = mat_obj['output']['RF_Waveforms_mT'][0][0].T
        avg_norm = np.mean(np.linalg.norm(rf_waveform, axis=0))
        return avg_norm

    def get_flip_angle_map(self, file_name):
        sel_mat_file = os.path.join(self.ddata, file_name)
        mat_obj = scipy.io.loadmat(sel_mat_file)
        kt_spoke_array = mat_obj['output']['Predicted_Solution'][0][0].T
        kt_spoke_array = convert_thomas_array_to_me(kt_spoke_array)

        return self.flip_angle_factor * kt_spoke_array


class StoreKtImage(KtImage):
    def __init__(self, ddata, ddest, coil_name, weird_rf_factor=1, flip_angle_factor=1, load_Q=False):
        super().__init__(ddata, ddest, coil_name, flip_angle_factor=flip_angle_factor,
                         weird_rf_factor=weird_rf_factor, load_Q=load_Q)

    def get_time_avg_SAR(self, file_name):
        container = self.mat_reader.read_Q_object()
        self.SAR_array = container['Q10g']
        rf_waveform = self.get_unique_pulse_settings(file_name)
        # nc -> n pulses, c coils
        # cdzyx -> c coils, d coils, zyx coordinates
        # dn -> d coils, n pulses
        # Returns n pulses by zyx coordinates
        time_avg_SAR = np.einsum("nc,cdzyx,dn->nzyx", rf_waveform.T.conjugate(), self.SAR_array, rf_waveform)
        head_sar_correction, avg_power_deposition = self.time_averaged_power_deposition_matrix(file_name)

        return time_avg_SAR, head_sar_correction, avg_power_deposition

    def store_time_avg_SAR(self, file_name):
        # This array does not use any normalization based on the power deposition matrix
        # This migt be needed though
        time_avg_SAR, head_sar_correction, avg_power_deposition = self.get_time_avg_SAR(file_name)
        np.save(self.path_sar_file + '_forward_power' + f'_{self.flip_angle_factor}', time_avg_SAR)
        np.save(self.path_sar_file + '_head_sar' + f'_{self.flip_angle_factor}', time_avg_SAR * head_sar_correction)
        return avg_power_deposition

    def store_flip_angle(self, file_name):
        kt_spoke_array = self.get_flip_angle_map(file_name)
        np.save(self.path_flip_angle_file + f'_{self.flip_angle_factor}', kt_spoke_array)


class VisualizeKtImage(KtImage):
    def __init__(self, ddata, ddest, coil_name, str_normalization, weird_rf_factor=1, flip_angle_factor=1):
        """

        :param ddata:
        :param ddest:
        :param coil_name:
        :param str_normalization:  Used to select a specific normalization. These images are stored with the StorekTimage class
        :param weird_rf_factor:
        :param flip_angle_factor:
        """
        super().__init__(ddata, ddest, coil_name, flip_angle_factor=flip_angle_factor, weird_rf_factor=weird_rf_factor)

        self.flipangle_map = None
        self.time_avg_sar = None

        flip_angle_file = self.path_flip_angle_file + f'_{self.flip_angle_factor}.npy'
        if os.path.isfile(flip_angle_file):
            self.flipangle_map = np.abs(np.load(flip_angle_file))

        sar_file_name = self.path_sar_file + str_normalization + f'_{self.flip_angle_factor}.npy'
        if os.path.isfile(sar_file_name):
            print('Loading ', sar_file_name)
            self.time_avg_sar = np.load(sar_file_name)
            self.n_spokes = self.time_avg_sar.shape[0]

        self.dindv_sar_spokes = os.path.join(self.ddest, f'individual_spoke_sar_{str_normalization}_{flip_angle_factor}.png')
        self.davg_sar = os.path.join(self.ddest, f'time_avg_sar_{str_normalization}_{flip_angle_factor}.png')
        self.dflip_angle_png = os.path.join(self.ddest, f'flip_angle_{str_normalization}_{flip_angle_factor}.png')

    def plot_flip_angle(self):
        assert self.flipangle_map is not None
        mid_slices = hplotf.get_all_mid_slices(self.flipangle_map[::-1], offset=MID_SLICE_OFFSET)
        fig_obj = hplotc.ListPlot([mid_slices], augm='np.abs', cbar=True, cbar_round_n=2, wspace=0.2,
                                  title=self.coil_plot_name, ax_off=True, cmap=COLOR_MAP, vmin=(0.75*self.flip_angle_factor, self.flip_angle_factor*1.25))
        fig_obj.figure.savefig(self.dflip_angle_png, bbox_inches='tight')

    def plot_avg_sar(self):
        assert self.time_avg_sar is not None
        # Average all the SAR spokes and plot it...
        temp_array = np.mean(self.time_avg_sar, axis=0)[::-1]
        array_slices = hplotf.get_all_mid_slices(temp_array, offset=MID_SLICE_OFFSET)
        vmax = np.max([np.max(np.abs(x)) for x in array_slices])  #???
        fig_obj = hplotc.ListPlot([array_slices], augm='np.abs', cbar=True, cbar_round_n=2, wspace=0.2,
                                  title=self.coil_plot_name, ax_off=True, cmap=COLOR_MAP, vmin=(0, vmax))
        fig_obj.figure.savefig(self.davg_sar, bbox_inches='tight')

    def plot_flip_angle_list(self):
        fa_list = []
        subtitle_list = []
        for i_file in self.output_design_files:
            avg_norm = self.get_avg_norm_waveform(i_file).round(2)
            flip_angle_3d = np.abs(self.get_flip_angle_map(i_file))
            axial_array = hplotf.get_all_mid_slices(flip_angle_3d, offset=MID_SLICE_OFFSET)[-1]
            fa_list.append(axial_array)
            subtitle_list.append(avg_norm)

        return fa_list, subtitle_list

    def report_metrics(self, pd_dataframe=None):
        # TODO: AAH heb je zo een mooie mask functie gemaakt.. ga je hiet hier hard-coden.. oh oh oh...
        coef_var_fa = hmetric.coefficient_of_variation(self.flipangle_map[self.thomas_mask_array == 1]) * 100
        rmse, nrmse = hmetric.normalized_rmse(self.flipangle_map[self.thomas_mask_array == 1], self.flip_angle_factor)
        # Get the original spoke file back...
        # This might not go that well with beta stuff...
        max_peak_sar = np.max(self.time_avg_sar.mean(axis=0).real)
        avg_sar = self.time_avg_sar[:, self.thomas_mask_array == 1].real.mean().real
        metric_dict = {'cov flip angle': coef_var_fa, 'RMMSE flip angle': rmse,
                       'NRMSE flip angle': nrmse, 'peak SAR': max_peak_sar,
                       'avg SAR': avg_sar}
        metric_dataframe = pd.DataFrame(metric_dict, index=[self.coil_plot_name])

        if pd_dataframe is not None:
            pd_dataframe = pd.concat([pd_dataframe, metric_dataframe])
        else:
            pd_dataframe = metric_dataframe

        return pd_dataframe

    def get_plot_sar_spokes(self):
        multi_sar_distr = []
        for i_spoke in range(self.n_spokes):
            temp_array = self.time_avg_sar[i_spoke]
            axial_slice = hplotf.get_all_mid_slices(temp_array[::-1], offset=MID_SLICE_OFFSET)[-1]
            # Using ::-1 to flip the orientation of the head
            multi_sar_distr.append(axial_slice[::-1])

        axial_avg_slice = np.mean(multi_sar_distr, axis=0)
        multi_sar_distr.append(axial_avg_slice)
        return multi_sar_distr



