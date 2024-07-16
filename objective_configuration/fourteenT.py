import os

"""
File contains the options and paths that we are going to explore
"""
username = os.environ.get('USER', os.environ.get('USERNAME'))
remote = False
if username != 'bugger':
    remote = True

if remote:
    # Remote duurt nogal lang om data over te zetten....
    # DPLOT = '/data/seb/paper/14T'
    DPLOT = '/home/bme001/20184098/data/boromir/seb/paper/paper/14T/'
    # DDATA = '/data/seb/14T'
    DDATA = '/home/bme001/20184098/data/boromir/seb/14T/'
else:
    DPLOT = '/home/bugger/Documents/paper/14T'
    # DDATA = '/home/bugger/Documents/data/14T'
    # Moved it to the HDD
    DDATA = '/media/bugger/MyBook/data/14T'

"""
Creating directories
"""

MID_SLICE_OFFSET = (-8, 0, 0)
# Plot directories
DPLOT_1KT_BETA_VOP = os.path.join(DPLOT, 'results_1kt_beta_vop')  # One spoke on VOP
DPLOT_KT_BETA_VOP = os.path.join(DPLOT, 'results_kt_beta_vop')  # Varying beta on SAR
DPLOT_KT_VOP = os.path.join(DPLOT, 'results_kt_vop')  # Individual spoke on SAR

DPLOT_1KT_BETA_POWER = os.path.join(DPLOT, 'results_1kt_beta_power')  # One spoke on POWER
DPLOT_KT_BETA_POWER = os.path.join(DPLOT, 'results_kt_beta_power')  # Varying beta on POWER
DPLOT_KT_POWER = os.path.join(DPLOT, 'results_kt_power')  # Individual spoke on POWER

DPLOT_FINAL = os.path.join(DPLOT, 'final_figures')

# Data directories containing results from simulations
DDATA_1KT_BETA_POWER = os.path.join(DDATA, 'results_1kt_beta_power')  # One spoke on POWER
DDATA_KT_BETA_POWER = os.path.join(DDATA, 'results_kt_beta_power')  # Varying beta on POWER
DDATA_KT_POWER = os.path.join(DDATA, 'results_kt_power')  # Individual spoke on POWER

DDATA_1KT_BETA_VOP = os.path.join(DDATA, 'results_1kt_beta_vop')  # One spoke on VOP
DDATA_KT_BETA_VOP = os.path.join(DDATA, 'results_kt_beta_vop')  # Varying beta on SAR
DDATA_KT_VOP = os.path.join(DDATA, 'results_kt_vop')  # Individual spoke on SAR


# We made this a global variable to play with variations and not overwrite previous results
#SUBDIR_OPTIM_SHIM = 'optim_shim_recalc_sar' # This was the old one
SUBDIR_OPTIM_SHIM = 'optim_shim_2' # Using this one for creating a plot for my thesis
SUBDIR_RANDOM_SHIM = 'random_shim_binned'
SUBDIR_OPTIM_B1 = 'optim_b1'
SUBDIR_OPTIM_SAR = 'optim_SAR'

"""
Defining constants..
"""

# Zo, dit moet toch overal hetzelfde..
Y_METRIC = 'peak_SAR'
X_METRIC = 'b1p_nrmse'
MAX_ITER = 25
TARGET_FLIP_ANGLE = 40  # = 40 degrees
TARGET_B1 = 1  # in muT
WEIRD_RF_FACTOR = 18
TARGET_DURATION = 0.768 * 1e-3  # = 0.768ms Derived from the length of Thomas' pulse
TARGET_DURATION_1KT = 0.166 * 1e-3  # = 0.768ms Derived from the length of Thomas' pulse
DT_KT = 6.4e-6  # in seconds
mean_b1 = TARGET_B1 * 1e-6  # T  (= 1muT)
gamma = 267.52218744 * 1e6  # rad/s/T
FA_rad = 3.141592653589793 / 180 * TARGET_FLIP_ANGLE  # rad
# Desired scaling of an RF pulse
RF_SCALING_FACTOR = FA_rad / (gamma * mean_b1 * TARGET_DURATION)
RF_SCALING_FACTOR_1KT = FA_rad / (gamma * mean_b1 * TARGET_DURATION_1KT)

# Create all these plotting directories...
temp_ddir_list = [DPLOT_KT_POWER,
                  DPLOT_KT_BETA_POWER,
                  DPLOT_1KT_BETA_POWER,
                  DPLOT_KT_VOP,
                  DPLOT_KT_BETA_VOP,
                  DPLOT_1KT_BETA_VOP,
                  DPLOT_FINAL]

for i_dir in temp_ddir_list:
    if not os.path.isdir(i_dir):
        os.makedirs(i_dir)

# File name of thomas his mask.. based one create_mask_thomas_roos.py
DMASK_THOMAS = os.path.join(DDATA, 'thomas_roos_mask.npy')

# Data directories containing preprocessed data
DDATA_POWER_DEPOS = os.path.join(DDATA, 'power_deposition_matrix')
if not os.path.isdir(DDATA_POWER_DEPOS):
    os.makedirs(DDATA_POWER_DEPOS)

"""
Coil based variables..
"""
ARRAY_SHAPE = (95, 87, 87)

COIL_NAME_ORDER = ['8 Channel Dipole Array 7T', '8 Channel Dipole Array',
                   '16 Channel Loop Dipole Array', '15 Channel Dipole Array',
                   '16 Channel Loop Array small', '8 Channel Loop Array big']

COIL_NAME_ORDER_TRANSLATOR = {
    '8 Channel Dipole Array 7T': '8D7T',
    '8 Channel Dipole Array': '8D',
    '16 Channel Loop Dipole Array': '8D8L',
    '15 Channel Dipole Array': '15D',
    '16 Channel Loop Array small': '16L',
    '8 Channel Loop Array big': '8L'}

COIL_NAME_ORDER_TRANSLATED = [COIL_NAME_ORDER_TRANSLATOR[x] for x in COIL_NAME_ORDER]

"""
Defining results

We have selected specific shim settings for sepcific results...
"""

OPTIMAL_KT_POWER = {'8 Channel Dipole Array 7T': 4,  #
                     '8 Channel Dipole Array': 12,  #
                     '16 Channel Loop Dipole Array': 6,  #
                     '15 Channel Dipole Array': 3,  #
                     '16 Channel Loop Array small': 10,  #
                     '8 Channel Loop Array big': 16}  #

OPTIMAL_KT_POWER_1kt = {'8 Channel Dipole Array 7T': 1,  #
                     '8 Channel Dipole Array': 1,  #
                     '16 Channel Loop Dipole Array': 1,  #
                     '15 Channel Dipole Array': 1,  #
                     '16 Channel Loop Array small': 1,  #
                     '8 Channel Loop Array big': 1}  #


OPTIMAL_KT_SAR = {'8 Channel Dipole Array 7T': 8,
                 '8 Channel Dipole Array': 19,
                 '16 Channel Loop Dipole Array': 11,
                 '15 Channel Dipole Array': 10,
                 '16 Channel Loop Array small': 7,
                 '8 Channel Loop Array big': 16}


OPTIMAL_KT_SAR_1kt = {'8 Channel Dipole Array 7T': 5,
                 '8 Channel Dipole Array': 5,
                 '16 Channel Loop Dipole Array': 10,
                 '15 Channel Dipole Array': 10,
                 '16 Channel Loop Array small': 6,
                 '8 Channel Loop Array big': 7}


# This one is updated for the recalculated SAR values
OPTIMAL_SHIM_POWER = {'8 Channel Dipole Array 7T': (21, 7),
                         '8 Channel Dipole Array': (15, 6),
                         '16 Channel Loop Dipole Array': (11, 6), # This one has recently been improved
                         '15 Channel Dipole Array': (4, 3),
                         '16 Channel Loop Array small': (21, 6),
                         '8 Channel Loop Array big': (8, 5)}

# This one is not made yet...
OPTIMAL_SHIM_SAR = {'8 Channel Dipole Array 7T': (1, 1),
                     '8 Channel Dipole Array': (1, 1),
                     '16 Channel Loop Dipole Array': (1, 1),
                     '15 Channel Dipole Array': (1, 1),
                     '16 Channel Loop Array small': (1, 1),
                     '8 Channel Loop Array big': (1, 1)}

"""
Creating colors...
"""

import matplotlib.pyplot as plt
cmap_str = 'Dark2'
_n_models = len(COIL_NAME_ORDER)
_plt_cm = plt.get_cmap(cmap_str)
_color_list = [_plt_cm(1. * i / (_n_models + 1)) for i in range(1, _n_models + 1)]
COLOR_DICT = {k: _color_list[ii] for ii, k in enumerate(COIL_NAME_ORDER)}

PLOT_LINEWIDTH = 2
COLOR_MAP = 'viridis'

"""
Create the different options that we used to plot stuff
"""
# Redo this
CALC_OPTIONS = [{'full_mask': True, 'type_mask': 'thomas_mask', 'objective_str': 'rmse_power'},
                {'full_mask': True, 'type_mask': 'thomas_mask', 'objective_str': 'rmse_sar'}]

for i_options in CALC_OPTIONS:
    full_mask = i_options['full_mask']
    type_mask_str = i_options['type_mask']
    objective_str = i_options['objective_str']
    full_mask_str = 'slice'
    if full_mask:
        full_mask_str = 'body'

    ddest = os.path.join(DPLOT, f'plot_{full_mask_str}_{type_mask_str}_{objective_str}')
    i_options['ddest'] = ddest

print(CALC_OPTIONS)
