"""
We know the current form of the signal model...

Lets check how it behaves.....

it shows that we have been off by a larger margin I guess
"""

import helper.array_transf as harray
import numpy as np
import helper.plot_class as hplotc
import matplotlib.pyplot as plt

from small_project.sinp3.signal_equation import *

"""
Typical example for a SE sequence
"""

fa_degree = np.arange(0, 360)
fa_radians = np.deg2rad(fa_degree)

TR_se = 10000  # msec
TE_se = 53  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 583  # msec
T2_fat = 46  # msec
T1_muscle = 1552  # msec
T2_muscle = 23  # msec

signal_fat = get_t2_signal(flip_angle=fa_radians, T1=T1_fat, T2=T2_fat, TE=TE_se, TR=TR_se)
signal_muscle = get_t2_signal(flip_angle=fa_radians, T1=T1_muscle, T2=T2_muscle, TE=TE_se, TR=TR_se)
simplified_signal = get_t2_signal_simplified(flip_angle=fa_radians)

# This shows that the equation we used actually shows a sin(alpha) dependence... and not a sin(alpha) ** 3
# Which is weird.. just weird.

plt.figure()
plt.plot(fa_degree, signal_muscle, 'b', label='muscle')
plt.plot(fa_degree, signal_fat, 'r', label='fat')
plt.plot(fa_degree, simplified_signal, 'k', label='paper')
plt.legend()


"""
Typical example for a SE sequence, comparing multiple refocussing pulses
"""

fa_degree = np.arange(0, 360)
fa_radians = np.deg2rad(fa_degree)

TR_se = 5000  # msec
TE_se = 53  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 583  # msec
T2_fat = 46  # msec
T1_muscle = 1552  # msec
T2_muscle = 23  # msec


plt.figure()
# Waarom tussen -pi...pi?

for irefocus in np.arange(0, 100, 1):
    temp_signal = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, T2=T2_fat, TE=TE_se, TR=TR_se, N=irefocus, beta=2 * fa_radians)
    if irefocus > 80:
        plt.plot(temp_signal, label=irefocus)
    else:
        plt.plot(temp_signal)
plt.legend()

"""
Typical example for a GE sequence

Looks weird, more nonlinear towards the sin**3 but nothing close.
"""

fa_degree = np.arange(0, 360)
fa_radians = np.deg2rad(fa_degree)

TR_se = 6.5  # msec
TE_se = 3  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 583  # msec
T2_fat = 46  # msec
T1_muscle = 1552  # msec
T2_muscle = 23  # msec

signal_fat = get_t2_signal(flip_angle=fa_radians, T1=T1_fat, T2=T2_fat, TE=TE_se, TR=TR_se)
signal_muscle = get_t2_signal(flip_angle=fa_radians, T1=T1_muscle, T2=T2_muscle, TE=TE_se, TR=TR_se)
simplified_signal = get_t2_signal_simplified(flip_angle=fa_radians)

# This shows that the equation we used actually shows a sin(alpha) dependence... and not a sin(alpha) ** 3
# Which is weird.. just weird.

plt.figure()
plt.twinx().plot(fa_degree, signal_muscle, 'b', label='muscle')
plt.twinx().plot(fa_degree, signal_fat, 'r', label='fat')
plt.twinx().plot(fa_degree, simplified_signal, 'k', label='paper')
plt.legend()


"""
Using a Bloch simulator because I dont trust all those derivations...
They must still be based on some small tip angle stupidity

Okay this has proven to be more dificult than I hoped.
"""

import bloch.rf_seq
import bloch.bloch
import bloch.pulse_seq_design

# I dont understand this........
Gz = 0.5  # 50mT/cm
Gz_max = 1 # 50 mT / cm / s
TH = 10  # 10 cm
gamma = 26747.52
gamma_bar = gamma / (2 * np.pi)
BW_rf = gamma_bar * Gz * TH
duration = 5e-3  # 5 ms
dt = 1e-6
flip_angle = np.pi/2
sinc_pulse = bloch.rf_seq.sinc_pulse(timebandwidth=BW_rf, flip_angle=flip_angle, duration=duration, dt=dt)
plt.plot(sinc_pulse)
grwform, indices = bloch.pulse_seq_design.generate_readout_gradient(Nf=200, fov_r=20, bwpp=BW_rf, g_max=Gz, s_max=Gz, dt=dt)
result = bloch.bloch.bloch(sinc_pulse, grwform, tp=dt, t1=1500, t2=46, df=0, dp=1, mode=1)

mx, my, mz = result
fig, ax = plt.subplots(3)
counter = 0
for iax in ax:
    iax.plot(result[counter])
    counter += 1


"""
Compare the LASI signal, the generic signal and the T2 one from Wang
"""

fa_degree = np.arange(0, 180)
fa_radians = np.deg2rad(fa_degree)

TR_se = 5000 * 1e-3
TE_se = 53 * 1e-3  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 583 * 1e-3
T2_fat = 46 * 1e-3


# Compare the t2_signal (Wang paper) with a specific (LASI) and generic (general) signal equation
# They overlap now, had trouble with a minus sign.
signal_fat = get_t2_signal(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, beta=np.pi, T2=None)
original_signal = get_t2_signal_LASI(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=None)
general_signal = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=None, N=1, beta=np.pi)

plt.figure()
plt.plot(original_signal, 'k.-', label='without T2, 180 and N=1')
plt.plot(general_signal, 'r*', label='without T2, generic signal equation')
plt.plot(signal_fat, 'b--', label='without T2, signal equation from Wang 2005')
plt.legend()
fa_degree = np.arange(0, 180)
fa_radians = np.deg2rad(fa_degree)

# Now check the influence of using T2.. in the LASI equation they neglect this.
general_signal1 = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, N=1, beta=np.pi)
general_signal2 = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, N=1, beta=2*fa_radians)

# plt.figure()
plt.plot(general_signal1, 'g.-', label='with T2, 180 refocus')
plt.plot(general_signal2, 'y*', alpha=0.5, label='with T2, factor 2 refocus')
plt.plot(np.sin(fa_radians), 'r--', alpha=0.5, label='sin(alpha)')
plt.legend()


"""
Now we are going to check the influence of the assumptions:

    * TR >> T1
    * T1 >> TE
"""

fa_degree = np.arange(0, 360)
fa_radians = np.deg2rad(fa_degree)

TR_se = 1000000  # dummy
TE_se = 1  # dummy
T1_fat = 1000  # dummy
T2_fat = 46

general_signal = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, N=1, beta=np.pi)
simplified_signal = get_t2_signal_simplified(flip_angle=fa_radians)

plt.figure()
plt.plot(fa_degree, general_signal, 'r*')
plt.plot(fa_degree, simplified_signal, 'k')


"""
Now we are going to immitate the signal by using rotation matrices...
"""



fa_degree = np.arange(0, 360)
fa_radians = np.deg2rad(fa_degree)

TR_se = 5000 * 1e-3
TE_se = 53 * 1e-3  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 583 * 1e-3
T2_fat = 46 * 1e-3

def get_rotated_signal(alpha, x0=None):
    if x0 is None:
        x0 = np.array([0, 0, 1]).reshape(3,1)

    # Not doing anything with phi atm.. dont know how to integrate over that one..
    res = harray.rot_y(2 * alpha) @ harray.rot_x(alpha) @ x0
    return res

general_signal = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, N=1, beta=np.pi)
simplified_signal = get_t2_signal_simplified(flip_angle=fa_radians)

plt.figure()
plt.plot(fa_degree, general_signal, 'r*')
plt.plot(fa_degree, simplified_signal, 'k')


"""
Now we are going to assume that T1 -> infty and T2 and well
"""

fa_degree = np.arange(0, 360)
fa_radians = np.deg2rad(fa_degree)

TR_se = 1000  # dummy
TE_se = 50  # dummy
T1_fat = 99999999999999  # dummy
T2_fat = 99999999999999

container = get_t2_signal_parts(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat)
signal_se, numerator_term_1, numerator_term_2, numerator, denominator_factor_1, denominator_factor_2, denominator = container
plt.plot(signal_se)
print('numerator part 1', numerator_term_1)
print('numerator part 2', numerator_term_2)
print('numerator', numerator)
print('denominator part 1', denominator_factor_1)
print('denominator part 2', denominator_factor_2)

general_signal = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, N=1, beta=np.pi)
simplified_signal = get_t2_signal_simplified(flip_angle=fa_radians)

plt.figure()
plt.plot(fa_degree, general_signal, 'r*', label='')
plt.plot(fa_degree, simplified_signal, 'k')
plt.legend()


"""
Use the paper refered to in the Wang 2005 paper: LARGE ANGLE SPIN-ECHO IMAGING
 
Here they also have a figure on how the signal intensity changes... Lets remake that one
Evertyhing is calculated correctly. So we have the correct implementation for both
the generic implementation and the specific (N=1, beta=180) one
"""

fa_degree = np.arange(0, 180)
fa_radians = np.deg2rad(fa_degree)
TR = 150
T1 = 360
TE = 10
signal_LASI = get_t2_signal_LASI(flip_angle=fa_radians, T1=T1, TR=TR, TE=TE)
signal_generic = get_t2_signal_general(flip_angle=fa_radians, T1=T1, TR=TR, TE=TE, beta=np.pi, N=1, T2=None)
plt.plot(signal_LASI, 'k')
plt.plot(signal_generic, 'r*')


"""
How did they get the sin ** 3..???
 I think I should use values that respect their assumptions TR >> T1 >> TE
"""

fa_degree = np.arange(0, 180)
fa_radians = np.deg2rad(fa_degree)

TR_se = 10
T1_fat = 10  # dummy
TE_se = 1
T2_fat = 46

signal_generic = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TR=TR_se, TE=TE_se, beta=np.pi, N=1, T2=None)
simplified_signal = get_t2_signal_simplified(flip_angle=fa_radians)

plt.plot(signal_generic, 'k')
plt.plot(simplified_signal)
plt.plot(np.sin(fa_radians))

"""
Try different tissues
"""

fa_degree = np.arange(0, 180)
fa_radians = np.deg2rad(fa_degree)

# T1 / T2 values https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/
TR_se = 5000
TE_se = 53  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 583
T2_fat = 46
T1_muscle = 1552
T2_muscle = 23
T1_bone_marrow = 548
T2_bone_marrow = 47

signal_fat = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, N=10, beta=np.pi)
signal_muscle = get_t2_signal_general(flip_angle=fa_radians, T1=T1_muscle, TE=TE_se, TR=TR_se, T2=T2_muscle, N=10, beta=np.pi)
signal_bone_marrow = get_t2_signal_general(flip_angle=fa_radians, T1=T1_bone_marrow, TE=TE_se, TR=TR_se, T2=T2_bone_marrow, N=10, beta=np.pi)

plt.figure()
plt.plot(signal_muscle, 'r*', label='muscle')
plt.plot(signal_fat, 'b.-', label='fat')
plt.plot(signal_bone_marrow, 'g*', label='bone marrow')
plt.plot(np.sin(fa_radians), 'k.', label='sin(alpha)')
plt.plot(np.sin(fa_radians) ** 3, 'k--', label='sin(alpha) ** 3')
plt.xlabel('flip angle (alpha)')
plt.title('SE signal with 180 degree refocussing pulse')
plt.legend()

"""
Now just try to use stuff with different TR/TE values
"""


fa_degree = np.arange(0, 180)
fa_radians = np.deg2rad(fa_degree)

# T1 / T2 values https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/
TR_se = 5000
TR = np.arange(1, 5000, 250)
TE_se = 53  # https://www.nature.com/articles/s41598-019-54880-x
TE = np.arange(1, 150, 1)
T1_fat = 583
T2_fat = 46
fat_TR_TE_variation = []
for i_TR in TR:
    temp_list = []
    for i_TE in TE:
        temp = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=i_TE, TR=i_TR, T2=T2_fat, N=1, beta=np.pi)
        temp_list.append(temp)
    fat_TR_TE_variation.append(temp_list)

fat_TR_TE_variation = np.array(fat_TR_TE_variation)
hplotc.SlidingPlot(fat_TR_TE_variation, ax_3d=True)

container = get_t2_signal_parts(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, beta=np.pi)
signal_se, numerator_term_1, numerator_term_2, numerator, denominator_factor_1, denominator_factor_2, denominator = container
plt.plot(signal_se)
print('numerator part 1', numerator_term_1)
print('numerator part 2', numerator_term_2)
print('numerator', numerator)
# print('denominator part 1', denominator_factor_1)
print('denominator part 2', denominator_factor_2)
# print('denominator', denominator)

plt.figure()
plt.plot(signal_muscle, 'r*', label='muscle')
plt.plot(signal_fat, 'b.-', label='fat')
plt.plot(signal_bone_marrow, 'g*', label='bone marrow')
plt.plot(np.sin(fa_radians), 'k.', label='sin(alpha)')
plt.plot(np.sin(fa_radians) ** 3, 'k--', label='sin(alpha) ** 3')
plt.xlabel('flip angle (alpha)')
plt.title('SE signal with 180 degree refocussing pulse')
plt.legend()



fa_degree = np.arange(0, 180)
fa_radians = np.deg2rad(fa_degree)

# T1 / T2 values https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/
TR_se = 5000
TE_se = 53  # https://www.nature.com/articles/s41598-019-54880-x
T1_fat = 583
T2_fat = 46
N = np.arange(1, 105)
temp_list = []
for i_N in N:
    temp = get_t2_signal_general(flip_angle=fa_radians, T1=T1_fat, TE=TE_se, TR=TR_se, T2=T2_fat, N=i_N, beta=np.pi)
    temp_list.append(temp)
temp_list = np.array(temp_list)
hplotc.SlidingPlot(temp_list, ax_3d=True)

"""
Now do the same for on obtained B1+ map
"""

import scipy.io
import helper.array_transf as harray
import tooling.shimming.b1shimming_single as mb1_single

ddata_flavio = '/home/bugger/Documents/data/test_clinic_registration/flavio_data/M01.mat'
A = scipy.io.loadmat(ddata_flavio)
b1_plus_array = np.moveaxis(A['Model']['B1plus'][0][0], -1, 0)
b1_plus_array = harray.scale_minmax(b1_plus_array, is_complex=True)
mask_array = A['Model']['Mask'][0][0]

# Visualize each line...
plt.plot(np.moveaxis(b1_plus_array[:, 128], 0, -1))

n_c, n_y, n_x = b1_plus_array.shape
y_center = n_y // 2
x_center = n_x // 2
shim_mask = np.zeros((n_y, n_x))
delta_x = int(0.1 * n_y)
shim_mask[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x] = 1
plt.imshow(shim_mask * np.abs(b1_plus_array[0]))

# Create relative phases...
b1_plus_array = b1_plus_array * np.exp(-1j * np.angle(b1_plus_array[0]))
b1_plus_array = harray.correct_mask_value(b1_plus_array, mask_array)

shimming_obj = mb1_single.ShimmingProcedure(b1_plus_array, shim_mask, relative_phase=True,
                                            str_objective='b1',
                                            debug=False)

x_opt, final_value = shimming_obj.find_optimum()
b1_plus_array = harray.apply_shim(b1_plus_array, cpx_shim=x_opt)
plt.imshow(np.abs(b1_plus_array) * (1 + shim_mask))


# Input should an already be shimmed b1 plus image...
n_y, n_x = b1_plus_array.shape
y_center, x_center = (n_y // 2, n_x // 2)
delta_x = int(0.1 * n_y)
# I know that nx and ny are equal...
# This should actually be equal to the region of the ... square above
min_coord = y_center - delta_x
max_coord = y_center + delta_x
x_sub = b1_plus_array[y_center - delta_x:y_center + delta_x, x_center - delta_x:x_center + delta_x]
x_mean = np.abs(x_sub.mean())

flip_angle = np.pi/2

plt.plot(np.abs(b1_plus_array[128, :]), 'r')
plt.vlines(x=max_coord, ymin=0, ymax=1)
plt.vlines(x=min_coord, ymin=0, ymax=1)

x_mean = 0.8 # This is done by inspection

# Taking the absolute values to make sure that values are between 0..1
# B1 plus interference by complex sum. Then using abs value to scale
target_angle = np.pi/2  #np.random.uniform(flip_angle - np.pi / 18, flip_angle + np.pi / 18)
flip_angle_map = np.abs(b1_plus_array) / x_mean * target_angle

plt.plot(np.rad2deg(flip_angle_map[128, :]), 'r')
plt.vlines(x=max_coord, ymin=0, ymax=90)
plt.vlines(x=min_coord, ymin=0, ymax=90)

# signal_map = get_t2_signal_general(flip_angle=flip_angle_map, T1=T1_fat, T2=T2_fat, TE=TE_se, TR=TR_se, beta=2*flip_angle_map)
signal_map = get_t2_signal_general(flip_angle=flip_angle_map, T1=T1_fat, T2=None, TE=TE_se, TR=TR_se, beta=np.pi, N=16)
signal_map_simple = get_t2_signal_simplified(flip_angle=flip_angle_map)

plt.plot(signal_map[128,:], 'k*')
plt.plot(signal_map_simple[128,:], 'r')
plt.plot(flip_angle_map[128,:], 'b')
plt.plot(np.sin(flip_angle_map[128, :]), 'g')

hplotc.ListPlot([signal_map_simple, signal_map, flip_angle_map], vmin=(0, np.pi), cbar=True)


"""
Found another plot in an old paper.... with that I can validate my equation for SE....
"""
import numpy as np
import helper.plot_class as hplotc

flip_angle = np.arange(0, 180, 1)
flip_angle = np.deg2rad(flip_angle)
refocus_pulse = 180
refocus_pulse = np.deg2rad(refocus_pulse)
TE = 80  # msec
TR = np.arange(0.1, 4, 0.05) * 1e3  #msec
# https://pubmed.ncbi.nlm.nih.gov/31049609/
# Magnetic resonance imaging T1 relaxation times for the liver, pancreas and spleen in healthy children at 1.5 and 3 tesla
# 1.5T
T1_spleen = 1172  #
T1_liver = 581  #
# https://reader.elsevier.com/reader/sd/pii/S2352621117300426?token=11D66B5B51C84E37599F2A7E8F501D6B08E864178CEC0EAC439515C5F8D3EC554CE1B00794AB7AB1368BEBF7F8538015&originRegion=eu-west-1&originCreation=20210531092809
# T2*-weighted imaging of the liver versus the spleen to assess hepatitisB-related cirrhosis and its ChildePugh cla
T2_spleen = 16.1
T2_liver = 16.1
signal_spleen = []
for i_TR in TR[::-1]:
    temp = get_t2_signal_general(flip_angle=flip_angle, T1=T1_spleen, T2=T2_spleen, beta=refocus_pulse, N=2, TR=i_TR, TE=TE)
    signal_spleen.append(temp)

signal_spleen = np.array(signal_spleen)

signal_liver = []
for i_TR in TR[::-1]:
    temp = get_t2_signal_general(flip_angle=flip_angle, T1=T1_liver, T2=T2_liver, beta=refocus_pulse, N=2, TR=i_TR, TE=TE)
    signal_liver.append(temp)

signal_liver = np.array(signal_liver)

test = np.tile(np.sin(flip_angle), len(TR)).reshape(len(TR), len(flip_angle))
hplotc.SlidingPlot(signal_liver , ax_3d=True)


#
x_range = np.arange(-10, 10, 0.01)
plt.plot(x_range, np.exp(x_range), 'k')
plt.plot(x_range, 1 + x_range, 'r')

# Approximation fourier series exp(x)
x_range = np.arange(-10, 10, 0.01)
fig, ax = plt.subplots(3)
ax[0].plot(x_range, np.exp(x_range), 'k')
ax[0].plot(x_range, 1 + x_range)
ax[0].plot(x_range, 1 + x_range + x_range ** 2)
ax[0].plot(x_range, 1 + x_range + x_range ** 2 + x_range ** 3)
# Same, different x axis
ax[1].plot(x_range, np.exp(x_range), 'k')
ax[1].plot(x_range, 1 + x_range)
ax[1].plot(x_range, 1 + x_range + x_range ** 2)
ax[1].plot(x_range, 1 + x_range + x_range ** 2 + x_range ** 3)
ax[1].set_xlim(-0.5, 0.5)
ax[1].set_ylim(0.5, 1.5)
#
ax[2].plot(x_range, np.exp(x_range), 'k')
ax[2].plot(x_range, 1 + x_range)
ax[2].plot(x_range, 1 + x_range + x_range ** 2)
ax[2].plot(x_range, 1 + x_range + x_range ** 2 + x_range ** 3)
ax[2].set_xlim(-0.1, 0.1)
ax[2].set_ylim(0.8, 1.2)

