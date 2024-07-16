import helper.reconstruction as hrecon
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import collections
import reconstruction.ReadPhyslog as read_phys
import reconstruction.ReadCpx as read_cpx


"""
Okay we got some unsorted cardiac data.. lets sort it with the physlog 
"""
#
# dmatthijs = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_01_19/derp_data'
# unsorted_100p_data = 'ca_19012022_1658096_8_2_transradial_no_trigger_100V4.mat'
# unsorted_100p_phys = 'SCANPHYSLOG_ca_19012022_1658096_8_1_transradial_no_trigger_100V4.log'
# unsorted_100p_sin = 'ca_19012022_1658096_8_2_transradial_no_trigger_100V4.sin'
# sorted_100p_cpx = 'ca_19012022_1658096_8_3_transradial_no_trigger_100V4.cpx'

# dmatthijs = '/media/bugger/MyBook/data/7T_scan/cardiac/2022_01_19/derp_data'
# unsorted_100p_data = 'ca_19012022_1652538_6_2_transradialdyn_100V4.mat'
# unsorted_100p_phys = 'SCANPHYSLOG_ca_19012022_1652538_6_1_transradialdyn_100V4.log'
# unsorted_100p_sin = 'ca_19012022_1652538_6_2_transradialdyn_100V4.sin'
# sorted_100p_cpx = 'ca_19012022_1652538_6_3_transradialdyn_100V4.cpx'

dmatthijs = '/media/bugger/B7DF-4571/2022_01_19/mat_data'
unsorted_100p_data = 'ca_19012022_1710203_10_2_transradialdyn_20V4.mat'
unsorted_100p_phys = 'SCANPHYSLOG_ca_19012022_1710203_10_1_transradialdyn_20V4.log'
unsorted_100p_sin = 'ca_19012022_1710203_10_2_transradialdyn_20V4.sin'


unsorted_100p_data = '/media/bugger/B7DF-4571/2022_03_05/V9_32575/matdata/v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.lab_data.mat'
unsorted_100p_phys = '/media/bugger/B7DF-4571/2022_03_05/V9_32575/SCANPHYSLOG_v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.log'
unsorted_100p_sin = '/media/bugger/B7DF-4571/2022_03_05/V9_32575/v9_05032022_1032049_8_1_transverse_radial_retrospectiveV4.sin'

# Read the data
phys_obj_100p = read_phys.ReadPhyslog(unsorted_100p_phys)
phys_obj_100p.visualize_label('ppu')
mat_obj = scipy.io.loadmat(unsorted_100p_data)['unsorted_data'][0][0]

# Get hyper parameters
n_coil = int(hrecon.get_key_from_sin_file(unsorted_100p_sin, 'nr_channel_names'))
n_radial_points, n_lines = mat_obj.shape
TR = float(hrecon.get_key_from_sin_file(unsorted_100p_sin, 'repetition_times')) * 10 ** -3
n_lines_per_coil = n_lines // n_coil
scan_time = np.arange(0, n_lines_per_coil) * TR
coil_lines = np.array([mat_obj[:, i::n_coil] for i in range(n_coil)])

# Get the trajectory for a single dynamic / acquisition
sin_file = unsorted_100p_sin
min_encoding = hrecon.get_key_from_sin_file(sin_file, 'non_cart_min_encoding_nrs').split()
max_encoding = hrecon.get_key_from_sin_file(sin_file, 'non_cart_max_encoding_nrs').split()
ovs = float(hrecon.get_key_from_sin_file(sin_file, 'non_cart_grid_overs_factor'))
width = int(hrecon.get_key_from_sin_file(sin_file, 'non_cart_grid_kernel_size'))
n_spoke = hrecon.get_n_spoke(sin_file)
trajectory = hrecon.get_trajectory_sin_file(sin_file)

# Get the ppu data
ppu_data = phys_obj_100p.phys_log_dict['ppu']
sample_rate_physlog = phys_obj_100p.sample_rate

stop_mark_index = np.argwhere([x == phys_obj_100p.stop_mark for x in phys_obj_100p.phys_log_dict['mark']])[0][0]
start_scan_index = int(stop_mark_index - (max(scan_time) / phys_obj_100p.sample_rate))
scan_ppu_data = np.array(ppu_data[start_scan_index:stop_mark_index])
time_ppu_data = np.arange(stop_mark_index - start_scan_index) * sample_rate_physlog

peak_indices, _ = scipy.signal.find_peaks(scan_ppu_data, distance=0.5*1/sample_rate_physlog)
plt.figure()
plt.plot(time_ppu_data, scan_ppu_data)
plt.scatter(time_ppu_data[peak_indices], scan_ppu_data[peak_indices], color='r')

# Define the cardiac phases and their time intervals
n_cardiac_phases = 30
for n_cardiac_phases in range(2, 20, 2):
    peak_binning = {}
    # These are the number of spokes per cardiac phase...
    for ii_phase in range(n_cardiac_phases):
        t0 = time_ppu_data[peak_indices][:-1] + ii_phase * np.diff(time_ppu_data[peak_indices]) / n_cardiac_phases
        t1 = time_ppu_data[peak_indices][:-1] + (ii_phase + 1) * np.diff(time_ppu_data[peak_indices]) / n_cardiac_phases
        # Now do some extra effort to get prepending or appending phases...
        avg_cardiac_phase = np.mean(np.diff(time_ppu_data[peak_indices]))
        avg_delta_phase = avg_cardiac_phase / n_cardiac_phases

        first_cardiac_phase = time_ppu_data[peak_indices[0]] - avg_delta_phase * (n_cardiac_phases - ii_phase)
        if first_cardiac_phase > 0:
            t0 = np.append(t0, first_cardiac_phase)
            t1 = np.append(t1, first_cardiac_phase + avg_delta_phase)

        last_cardiac_phase = time_ppu_data[peak_indices[-1]] + avg_delta_phase * (ii_phase + 1)
        if last_cardiac_phase < max(scan_time):
            t0 = np.append(t0, last_cardiac_phase - avg_delta_phase)
            t1 = np.append(t1, last_cardiac_phase)
        #
        # plt.plot(time_ppu_data, scan_ppu_data)
        # plt.scatter(time_ppu_data[peak_indices], scan_ppu_data[peak_indices], color='r')
        # for x0, y0 in zip(t0, t1):
        #     plt.vlines(x=x0, ymin=-1000, ymax=4000, colors='k')
        #     plt.vlines(x=y0, ymin=-1000, ymax=4000, colors='k')

        intervals_of_profiles = list(zip((t0 / TR).astype(int), (t1 / TR).astype(int)))
        # Collect the profiles...
        for i_interval in intervals_of_profiles:
            start_index, end_index = i_interval
            phase_name = f'phase_{str(ii_phase).zfill(2)}'
            peak_binning.setdefault(phase_name, [])
            spoke_indices = list(range(start_index, end_index))
            peak_binning[phase_name].extend(spoke_indices)

    # Validate binning.
    import matplotlib
    cmap = matplotlib.cm.get_cmap('plasma', lut=n_cardiac_phases)

    fig, ax = plt.subplots()
    fig.suptitle(n_cardiac_phases)

    fig_traj, ax_traj = plt.subplots()
    fig_traj.suptitle(str(n_cardiac_phases))

    max_plot_value = max(abs(scan_ppu_data))
    ax.plot(time_ppu_data, scan_ppu_data)
    ax.scatter(time_ppu_data[peak_indices], scan_ppu_data[peak_indices], color='r')
    acquisition_angles = np.array(hrecon.get_angle_spokes(n_spoke))
    trajectory = hrecon.get_trajectory_sin_file(sin_file)
    for i_card in range(n_cardiac_phases):

        phase_name = f'phase_{str(i_card).zfill(2)}'
        temp = np.array(peak_binning[phase_name])
        for ii, i_x in enumerate(temp * TR):
            ax.vlines(x=i_x, ymin=-max_plot_value, ymax=max_plot_value, colors=cmap(i_card), zorder=-1)

    temp_traj = trajectory[temp % n_spoke][:, ::50]
    ax_traj.scatter(temp_traj[:, :, 0], temp_traj[:, :, 1])
    # fig_traj.suptitle(phase_name + '--' + str(n_cardiac_phases))

# Klopt het dat.... nscan = ndyn * nspoke? Yes
    # Hoeveel spokes zijn er... hoe veel tijd kost dit.. en hoe zit dit met de cardiac phase/hartslag


    # Lets reconstruct... one just thing..
    # I do need to get the spokes per acq line though
    # I dont have all the spokes right now.... is that .. OK?
    # Lets just get it on with it.
    import sigpy
    temp_phase = []
    for i_phase in range(n_cardiac_phases):
        phase_name = f'phase_{str(i_phase).zfill(2)}'
        phase_00 = coil_lines[:, :, peak_binning[phase_name]]
        phase_00_spoke_index = np.array(peak_binning[phase_name]) % n_spoke
        trajectory_00 = trajectory[phase_00_spoke_index, :, :]

        temp = []
        for i in range(n_coil):
            dcf = np.sqrt(trajectory_00[:, :, 0] ** 2 + trajectory_00[:, :, 1] ** 2)
            temp_img = sigpy.nufft_adjoint(phase_00[i].T * dcf, coord=trajectory_00, width=width, oversamp=ovs)
            temp.append(temp_img)

        temp_phase.append(temp)

    target_shape = np.max([list(np.array(x).shape) for x in temp_phase], axis=0)
    temp_phase = [harray.to_shape(np.array(x), target_shape) for x in temp_phase]
    temp_phase22 = [np.abs(x).sum(axis=0) for x in temp_phase]
    test = np.array(temp_phase22)
    test.shape
    zz = np.abs(test).sum(axis=1)
    hplotc.SlidingPlot(test)

cpx_obj = read_cpx.ReadCpx(os.path.join(dmatthijs, sorted_100p_cpx))
cpx_array = cpx_obj.get_cpx_img()
ww = np.squeeze(np.abs(cpx_array).sum(axis=0))

# We got a result... lets push that through a DL recon model
# Dit ding is nu 3.3 Gb... tering groot.
# Kan wel wat data missen.. en als int8 of int 16 opslaan. Zal we schelen
ddest = '/media/bugger/MyBook/data/7T_data/sorted_untriggered_data'
np.save(os.path.join(ddest, 'matthijs_100p_untriggered.npy'), test)


