import helper.reconstruction as hrecon
import matplotlib
import helper.array_transf as harray
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sigpy


class GetCardiacSignal:
    def __init__(self, unsorted_file, sin_file, mat_key='unsorted_data'):
        self.sin_file = sin_file
        self.unsorted_radial_data = scipy.io.loadmat(unsorted_file)[mat_key][0][0]

        self.n_coil = int(hrecon.get_key_from_sin_file(self.sin_file, 'nr_channel_names'))
        self.TR = float(hrecon.get_key_from_sin_file(self.sin_file, 'repetition_times')) * 10 ** -3

        # Reshapes the data to (n_coils, n_points, n_profiles_per_coil)
        self.radial_profiles_per_coil = np.array([self.unsorted_radial_data[:, i::self.n_coil] for i in range(self.n_coil)])
        self.n_radial_points, n_total_profiles = self.unsorted_radial_data.shape
        self.n_profiles_per_coil = n_total_profiles // self.n_coil
        self.scan_time = np.arange(0, self.n_profiles_per_coil) * self.TR
        self.center_index = self.n_radial_points // 2

    def get_averaged_center_signal(self, sel_coil=None, complex_part=None):
        # This gets the signal from the center of the image. As shown by `visualize_signal`
        running_mean_width = int(self.n_profiles_per_coil * 0.05)
        running_mean_data = harray.running_mean(self.radial_profiles_per_coil[sel_coil, self.center_index], running_mean_width)
        if complex_part == 'imag':
            running_mean_data = running_mean_data.imag
        elif complex_part == 'real':
            running_mean_data = running_mean_data.real
        else:
            print('derp')

        return running_mean_data

    def visualize_averaged_center_signal(self, n_coil=8):
        # With this we can see the center signal....
        sub_sel_coil = np.arange(-n_coil, 0)
        fig, ax = plt.subplots(len(sub_sel_coil))
        fig.suptitle('Central signal intensity per coil over time of radial spokes')
        for i_coil in sub_sel_coil:
            running_mean_data = harray.running_mean(self.radial_profiles_per_coil[i_coil, self.center_index], int(self.n_profiles_per_coil * 0.05))
            ax[abs(i_coil) - 1].plot(self.scan_time, running_mean_data.real, 'b', label='real')
            ax[abs(i_coil) - 1].plot(self.scan_time, running_mean_data.imag, 'r', label='imaginary')
            ax[abs(i_coil) - 1].legend()
            ax[abs(i_coil) - 1].set_title(i_coil)

        return fig


class SortCardiacRadialData:
    def __init__(self, unsorted_file, sin_file, n_cardiac_phases, cardiac_signal, sampling_freq,
                 mat_key='unsorted_data', **kwargs):
        # unsorted_file - the file that loads the unsorted radial data
        # sin_file - associated with the radial_coil_data acquisition that has all the scan parameters
        # n_cardiac_phases - number that divided the cardiacp hases
            # cardiac_signal - if not None should be a 1D array that contains the .. in the domain of the acquisition
            # This might be dificult... since we are defining a lot of stuff here

        self.sin_file = sin_file
        self.unsorted_radial_data = scipy.io.loadmat(unsorted_file)[mat_key][0][0]
        self.cardiac_signal = cardiac_signal
        self.n_cardiac_phases = n_cardiac_phases

        self.n_coil = int(hrecon.get_key_from_sin_file(self.sin_file, 'nr_channel_names'))
        self.TR = float(hrecon.get_key_from_sin_file(self.sin_file, 'repetition_times')) * 10 ** -3

        self.golden_angle = kwargs.get('golden_angle', False)

        # Reshapes the data to (n_coils, n_points, n_profiles_per_coil)
        self.radial_profiles_per_coil = np.array([self.unsorted_radial_data[:, i::self.n_coil] for i in range(self.n_coil)])
        self.n_radial_points, n_total_profiles = self.unsorted_radial_data.shape
        self.n_profiles_per_coil = n_total_profiles // self.n_coil

        self.ovs = float(hrecon.get_key_from_sin_file(sin_file, 'non_cart_grid_overs_factor'))
        self.width = int(hrecon.get_key_from_sin_file(sin_file, 'non_cart_grid_kernel_size'))
        # The number of spokes used in a single acquisition
        self.n_spoke = hrecon.get_n_spoke(sin_file)
        # The trajectory for a single acquisition
        self.trajectory = hrecon.get_trajectory_sin_file(sin_file, golden_angle=self.golden_angle)

        self.binning_obj = SignalBinning(signal=cardiac_signal, sampling_freq=sampling_freq, n_bins=n_cardiac_phases)
        self.binned_indices = self.binning_obj.time_intervals_to_index(self.TR)

    def visualize_trajectory(self, bin_number, stepsize=50):
        fig_traj, ax_traj = plt.subplots()
        fig_traj.suptitle(str(bin_number))
        bin_name = f'bin_{str(bin_number).zfill(2)}'
        temp = np.array(self.binned_indices[bin_name])
        temp_traj = self.trajectory[temp % self.n_spoke][:, ::stepsize]
        ax_traj.scatter(temp_traj[:, :, 0], temp_traj[:, :, 1])
        return fig_traj

    def reconstruct_cardiac_cine(self):
        # Lets reconstruct... one just thing..
        # I do need to get the spokes per acq line though
        # I dont have all the spokes right now.... is that .. OK?
        # Lets just get it on with it.

        temp_phase = []
        for i_phase in range(self.n_cardiac_phases):
            i_phase_name = f'bin_{str(i_phase).zfill(2)}'
            i_phase_indices = np.array(self.binned_indices[i_phase_name])
            i_phase_profiles = self.radial_profiles_per_coil[:, :, i_phase_indices]
            # We have done a dynamic study here... therefore we use modulo n_spoke
            # The total number of profiles is equal to the number of dynamics times the number of spokes..
            i_phase_spoke_index = i_phase_indices % self.n_spoke
            i_trajectory = self.trajectory[i_phase_spoke_index, :, :]

            temp = []
            for i in range(self.n_coil):
                dcf = np.sqrt(i_trajectory[:, :, 0] ** 2 + i_trajectory[:, :, 1] ** 2)
                temp_img = sigpy.nufft_adjoint(i_phase_profiles[i].T * dcf, coord=i_trajectory, width=self.width, oversamp=self.ovs)
                temp.append(temp_img)

            temp_phase.append(temp)

        target_shape = np.max([list(np.array(x).shape) for x in temp_phase], axis=0)
        reconstructed_image = np.array([harray.to_shape(np.array(x), target_shape) for x in temp_phase])
        return reconstructed_image


class SignalBinning:
    def __init__(self, signal, sampling_freq, n_bins=30, **kwargs):
        self.signal_length = len(signal)
        self.signal_time = np.arange(self.signal_length) * sampling_freq

        self.signal = signal
        self.sampling_freq = sampling_freq
        self.n_bins = n_bins

        distance_peaks = kwargs.get('distance', 0.8 * 1 / sampling_freq)
        self.peak_indices = self.get_signal_peaks(signal, distance=distance_peaks)
        if len(self.peak_indices) > 30:
            print("We are getting this amount of peaks: ", len(self.peak_indices))
        self.binned_signal = self.get_signal_bins()

    def get_signal_peaks(self, signal, distance, **kwargs):
        # Returns the peaks of the given signal
        # kwargs contains options for the peak detection
        peak_indices, _ = scipy.signal.find_peaks(signal, distance=distance, **kwargs)
        return peak_indices

    def get_signal_bins(self):
        # Performing binning...
        binned_signal = {}
        # We are going to loop over al the bins...  and store the time intervals
        # that satisfy these bins
        for ii_bin in range(self.n_bins):
            signal_start = self.signal_time[self.peak_indices][:-1]
            signal_difference = np.diff(self.signal_time[self.peak_indices]) / self.n_bins
            t0 = signal_start + ii_bin * signal_difference
            t1 = signal_start + (ii_bin + 1) * signal_difference

            # Now do some extra effort to get prepending or appending phases...
            avg_peak_duration = np.mean(np.diff(self.signal_time[self.peak_indices]))
            avg_bin_duration = avg_peak_duration / self.n_bins

            t1_first_cardiac_phase = self.signal_time[self.peak_indices[0]]
            first_cardiac_phase = t1_first_cardiac_phase - avg_bin_duration * (self.n_bins - ii_bin)
            if first_cardiac_phase > 0:
                t0 = np.append(t0, first_cardiac_phase)
                t1 = np.append(t1, first_cardiac_phase + avg_bin_duration)

            t0_last_cardiac_phase = self.signal_time[self.peak_indices[-1]]
            last_cardiac_phase = t0_last_cardiac_phase + avg_bin_duration * (ii_bin + 1)
            if last_cardiac_phase < max(self.signal_time):
                t0 = np.append(t0, last_cardiac_phase - avg_bin_duration)
                t1 = np.append(t1, last_cardiac_phase)

            bin_interval_list = zip(t0, t1)
            # Collect the profiles...
            for i_interval in bin_interval_list:
                start_index, end_index = i_interval
                bin_name = f'bin_{str(ii_bin).zfill(2)}'
                binned_signal.setdefault(bin_name, [])
                # For now we will just store/return the time interval
                # We used to translate these timings to spoke-indices.
                binned_signal[bin_name].extend((start_index, end_index))

        return binned_signal

    def time_intervals_to_index(self, new_sampling_frequency=None):
        if new_sampling_frequency is None:
            new_sampling_frequency = self.sampling_freq

        binned_index = {}
        for k, v in self.binned_signal.items():
            # Each binned index will contain the full range of indices in there.
            temp = []
            indices = (np.array(v).reshape((-1, 2)) / new_sampling_frequency).astype(int)
            for ix0, ix1 in indices:
                temp.extend(list(range(ix0, ix1)))

            binned_index[k] = np.array(temp)
        return binned_index

    def visualize_cardiac_bins(self):
        # Validate binning.
        cmap = matplotlib.cm.get_cmap('plasma', lut=self.n_bins)
        fig, ax = plt.subplots()
        fig.suptitle(f'Number of bins used {self.n_bins}')

        ax.plot(self.signal_time, self.signal)
        ax.scatter(self.signal_time[self.peak_indices], self.signal[self.peak_indices], color='r')
        for ii_bin in range(self.n_bins):
            bin_name = f'bin_{str(ii_bin).zfill(2)}'
            temp = np.array(self.binned_signal[bin_name])
            for i_interval in temp.reshape((-1, 2)):
                ax.axvspan(i_interval[0], i_interval[1], color=cmap(ii_bin), zorder=-1)

