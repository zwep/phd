import reconstruction.ReadCpx as read_cpx
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.plot_fun as hplotf
import numpy as np
import os
import sys
import scipy.optimize


class ShimmingProcedure:
    def __init__(self, input_array, mask, amp_shim=False, str_objective='b1', opt_method=None,
                 relative_phase=False, **kwargs):
        # Relative phase is to point out IF the input array is relative in its phase component.
        # It doesnt change anything to the input array...

        # List of possible optimization methods
        self.opt_method_options = ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG",
                                   "L-BFGS-B", "TNC", "COBYLA", "SLSQP", "trust, constr",
                                   "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]

        n_chan = input_array.shape[0]
        A_plus = input_array.reshape(n_chan, -1).T
        # self.input_array = input_array
        self.mask_ind = np.where(mask.ravel() == 1)[0]
        self.A_r_plus = A_plus[self.mask_ind, :]

        self.objective_dict = self.get_objective_dict_single()
        self.str_objective = str_objective
        self.objective = self.objective_dict.get(str_objective, None)

        if self.objective is None:
            print('Wrong objective chosen')
            print('Pick from ', self.objective_dict.keys())
            return

        self.relative_phase = relative_phase  # kwargs.get('relative_phase', False)
        # Set a starting distribution
        # Can also be used to test several objective functions in a sequential mode
        initial_shim = kwargs.get('initial_shim', None)

        # If we make use of a relative phase, we set the first coil to 1 amp and 0 phase
        # It is left out of the calculations
        if self.relative_phase:
            n_chan -= 1

        if initial_shim is None:
            initial_shim = self.get_random_amp_phase(n_chan, amp_shim=amp_shim)

        # Why did I do this..? Why reshape to -1, 1..?
        self.initial_shim = np.concatenate([initial_shim]) #.reshape((-1, 1))

        self.opt_method = opt_method
        # Use amplitude shimming
        self.amp_shim = amp_shim

        # For these optimization methods we can supply bounds
        if self.opt_method in ["L-BFGS-B", "TNC", "SLSQP", "Powell"]:
            self.bounds = self.get_bounds(n_chan, amp_shim=amp_shim)
        else:
            self.bounds = None

    def find_optimum(self, objective=None, opt_method=None, maxiter=250, verbose=False):
        if opt_method is None:
            opt_method = self.opt_method
        if objective is None:
            objective = self.objective

        x_opt = scipy.optimize.minimize(fun=objective, x0=self.initial_shim, tol=1e-8,
                                        method=opt_method, bounds=self.bounds,
                                        options={"maxiter": maxiter,
                                                 "disp": verbose}).x
        # I think this value can also be gotten from the minimize object... Quite sure of it even
        final_value = self.objective(x_opt)
        cpx_shim = self.to_complex(x_opt, amp_shim=self.amp_shim)
        if self.relative_phase:
            cpx_shim = np.insert(cpx_shim, 0, 1)
        return cpx_shim, final_value

    @staticmethod
    def get_bounds(n_chan, amp_shim):
        phase_bound = [(-np.pi, np.pi) for _ in range(n_chan)]
        amp_bound = []
        if amp_shim:
            amp_bound = [(0, 1) for _ in range(n_chan)]

        return amp_bound + phase_bound

    def get_objective_dict_single(self):
        objective_dict = {'b1': self.b1_in_roi_single,
                          'minb1': self.min_b1_in_roi_single,
                          'maxb1': self.max_b1_in_roi_single,
                          'homin': self.ho_min_roi_single,
                          'flip_angle': self.flip_angle_std,
                          'signal_ge': self.signal_ge_angle_std,
                          'signal_se': self.signal_se_angle_std}

        return objective_dict

    @staticmethod
    def get_random_amp_phase(n_chan, amp_shim=False):
        # Generated random phase or random phase + random amp. Based on the amount of channels are given

        random_amp = np.random.uniform(0, 1, size=n_chan)
        random_phase = np.random.uniform(-np.pi, np.pi, size=n_chan)
        # random_phase = np.zeros(n_chan)

        if amp_shim:
            initial_shim = np.concatenate([random_amp, random_phase])
        else:
            initial_shim = np.concatenate([random_phase])

        return initial_shim

    @staticmethod
    def to_complex(x, amp_shim=False):
        # Create a complex vector from x for computations
        # First half is always the amplitude, secnod half is always the phase
        if amp_shim:
            init_amp, init_phase = np.hsplit(x, 2)
        else:
            init_phase = x
            init_amp = np.ones(len(init_phase))

        return init_amp * np.exp(1j * init_phase)

    """
    Below are the different optimization criteria
    """
    def max_b1_in_roi_single(self, x):
        # Convert to complex values
        cpx_x = self.to_complex(x, self.amp_shim)
        if self.relative_phase:
            cpx_x = np.insert(cpx_x, 0, 1)
        temp_A = self.A_r_plus.dot(cpx_x)
        return -np.max(np.abs(temp_A))

    def b1_in_roi_single(self, x):
        A_2 = np.matmul(self.A_r_plus.T, self.A_r_plus)
        cpx_x = self.to_complex(x, self.amp_shim)
        if self.relative_phase:
            cpx_x = np.insert(cpx_x, 0, 1)
        temp_A = cpx_x.T.dot(A_2).dot(cpx_x)
        return -np.abs(temp_A)

    def min_b1_in_roi_single(self, x):
        cpx_x = self.to_complex(x, self.amp_shim)
        if self.relative_phase:
            cpx_x = np.insert(cpx_x, 0, 1)

        temp_A = self.A_r_plus.dot(cpx_x)
        return -np.min(np.abs(temp_A))

    def ho_min_roi_single(self, x):
        cpx_x = self.to_complex(x, self.amp_shim)
        if self.relative_phase:
            cpx_x = np.insert(cpx_x, 0, 1)
        temp_A = self.A_r_plus.dot(cpx_x)
        return -np.std(np.abs(temp_A)) / np.mean(np.abs(temp_A))

    def flip_angle_std(self, x):
        # Calculates the relative standard deviation (relative to mean)
        # on a single set of coils (only receive coils, or only transmit coils)
        cpx_x = self.to_complex(x, self.amp_shim)
        if self.relative_phase:
            cpx_x = np.insert(cpx_x, 0, 1)
        temp_A = self.A_r_plus.dot(cpx_x)
        signal = np.abs(temp_A)
        return -np.mean(signal) / np.std(signal)

    def signal_ge_angle_std(self, x):
        # Calculate approximated signal of gradient echo...
        cpx_x = self.to_complex(x, self.amp_shim)
        if self.relative_phase:
            cpx_x = np.insert(cpx_x, 0, 1)
        temp_A = self.A_r_plus.dot(cpx_x)
        signal = np.sin(np.abs(temp_A))
        return -np.mean(signal) / np.std(signal)

    def signal_se_angle_std(self, x):
        # Calculate approximated signal of gradient echo...
        cpx_x = self.to_complex(x, self.amp_shim)
        if self.relative_phase:
            cpx_x = np.insert(cpx_x, 0, 1)

        # This is already the maskeds stuff...
        temp_A = self.A_r_plus.dot(cpx_x)
        mean_value = np.mean(np.abs(temp_A))
        scale_factor = np.pi/2 * (1/mean_value)
        signal = np.sin(scale_factor * np.abs(temp_A)) ** 3
        return -np.mean(signal) / np.std(signal)


if __name__ == "__main__":

    # Load data
    # b1_shim_data = '/media/bugger/MyBook/data/7T_scan/prostate/2021_01_18/V9_16605/v9_18012021_0924085_3_3_b1shimseriesV4.cpx'
    # Vreemd wormpje...
    # b1_shim_data = '/media/bugger/MyBook/data/7T_scan/prostate/2020_05_27/V9_10168/v9_27052020_1546465_6_2_b1shimseriesV4.cpx'
    # b1_shim_data = '/media/bugger/MyBook/data/7T_scan/prostate/2020_07_01/V9_11279/v9_01072020_1704253_6_3_b1shimseriesV4.cpx'
    b1_shim_data = '/media/bugger/MyBook/data/7T_scan/prostate/2021_02_11/V9_17048/v9_11022021_1629256_2_3_b1shimseriesV4.cpx'
    read_cpx_obj = read_cpx.ReadCpx(b1_shim_data)
    img_data = read_cpx_obj.get_cpx_img()
    img_data = np.squeeze(np.rot90(img_data, axes=(-2, -1)))

    tx_data = img_data.sum(axis=0)
    rx_data = np.abs(img_data).sum(axis=1)

    tx_relative_phase = np.exp(-1j * (np.angle(img_data[:, 0:1])))
    rx_relative_phase = np.exp(-1j * (np.angle(img_data[0:1, :])))

    hplotc.ListPlot(np.abs(img_data).sum(axis=0).sum(axis=0), augm='np.abs', title='no shim used')
    # Plotting al channels..
    hplotc.ListPlot(img_data, augm='np.abs', title='overview of channels', vmin=(0, 1e5))
    hplotc.ListPlot(img_data, augm='np.angle', title='overview of channels', vmin=(0, 1e5))
    # Plotting TX data
    hplotf.plot_3d_list([tx_data], augm='np.abs', title='tx data')
    hplotf.plot_3d_list([tx_data * np.exp(-1j * np.angle(tx_data[0]))], augm='np.angle', title='tx data')
    hplotf.plot_3d_list([rx_data], augm='np.abs', title='rx data')
    # Plotting TX relative data
    hplotc.ListPlot(tx_relative_phase, augm='np.angle', title='tx phase first coil')
    hplotc.ListPlot((img_data * tx_relative_phase), augm='np.angle', title='tx relative phase to first coil')
    hplotc.ListPlot([(img_data * tx_relative_phase).sum(axis=0)], augm='np.angle', title='summed relative phase to first coil')
    # Plotting RX relative data
    hplotc.ListPlot(rx_relative_phase, augm='np.angle', title='')
    hplotc.ListPlot((img_data * rx_relative_phase), augm='np.angle')
    hplotc.ListPlot((img_data * rx_relative_phase).sum(axis=1), augm='np.angle')

    # # Startin shimming...
    mask_handle = hplotc.MaskCreator(tx_data * tx_relative_phase[:, 0])
    shimming_obj = ShimmingProcedure(tx_data * tx_relative_phase[:, 0], mask_handle.mask, str_objective='b1', relative_phase=True)
    x_opt, final_value = shimming_obj.find_optimum()
    complex_coils_shimmed = harray.apply_shim(tx_data * tx_relative_phase[:, 0], cpx_shim=x_opt)
    hplotc.ListPlot([complex_coils_shimmed, np.abs(tx_data).sum(axis=0)], augm='np.abs')
