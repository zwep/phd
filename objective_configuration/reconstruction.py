import os
import sys

"""

Storing model names now goes under

_parallel_discrete

Because there we train with separate train_25, train_50, train_75 and train_100 directories

First we tried 

_parallel

Where we used a mixed data set (hence parallel.. somehow) AND we used to occupy the lists OR filename_lists in the config file
However, this turned out to be rubbish...

Before that I ran stuff with

_sequential

Because we first trained on 2ch, then continued on transverse, and so on..


1075 train
112 val
156 test

"""


username = os.environ.get('USER', os.environ.get('USERNAME'))

DMAPPING_FBPCONVNET = DWEIGHTS_FBPCONVNET = None

# HPC
if username == '20184098':
    DDATA = '/home/bme001/20184098/data/direct'
    # Location of the radial kspace with individual spoke dataset
    DDATA_spoked = None
    # Location of the sin-files to create the individual spoke dataset
    DDATA_sin = None
    DRESULT = DPLOT = '/home/bme001/20184098/data/paper/reconstruction/results'
    # This is the location where we store all the trained models..
    DMODEL = '/home/bme001/20184098/model_run/direct'
    # Location of pretrained models from DIRECT
    DPRETRAINED = '/home/bme001/20184098/data/pretrained_networks/direct'
    # Inference result storage
    DRESULT_INFERENCE = '/home/bme001/20184098/data/paper/reconstruction/inference'
    # This one is not copied yet...
    DINFERENCE = '/home/bme001/20184098/data/direct_inference'
    DRESULT_INFERENCE_png = '/home/bme001/20184098/data/paper/inference_png'
# UMC
elif username == 'sharreve':
    # Location of the radial kspace  dataset
    DRECON = '/local_scratch/sharreve/paper/reconstruction'
    DDATA = '/local_scratch/sharreve/mri_data/cardiac_full_radial'
    DDATA_SYNTH = '/home/sharreve/local_scratch/mri_data/cardiac_synth_7T/direct_synth'
    # # Location of the radial kspace with individual spoke dataset
    # DDATA_spoked = '/local_scratch/sharreve/mri_data/cardiac_full_radial_spoked'
    # # Location of the sin-files to create the individual spoke dataset
    # DDATA_sin = '/local_scratch/sharreve/mri_data/cardiac_full_radial_sin'
    DRESULT = DPLOT = os.path.join(DRECON, 'results')
    # This is the location where we store all the trained models..
    DMODEL = '/home/sharreve/local_scratch/model_run/direct'
    # Location of pretrained models from DIRECT
    DPRETRAINED = '/home/sharreve/local_scratch/mri_data/pretrained_networks/direct'
    # Inference result storage
    DRESULT_INFERENCE = os.path.join(DRECON, 'inference')
    # DINFERENCE = '/local_scratch/sharreve/mri_data/cardiac_radial'
    DINFERENCE = '/local_scratch/sharreve/mri_data/cardiac_radial_inference'
    DRESULT_INFERENCE_png = '/local_scratch/sharreve/paper/reconstruction/inference_png'
    # # We have also retrospective data..
    DRETRO = '/local_scratch/sharreve/mri_data/cardiac_radial_inference_retrospective'
    DRESULT_RETRO = '/local_scratch/sharreve/paper/reconstruction/retro'
elif username == 'bugger':
    # Here is the mapping of the weights obtained by the FBPConvNet github page
    DMODEL = None
    DPRETRAINED = None
    DINFERENCE = '/home/bugger/Documents/data/paper/reconstruction/inference'
    DRETRO = DRESULT_RETRO = DDATA = DDATA_spoked = DDATA_sin = None
    DRECON = '/home/bugger/Documents/paper/reconstruction'
    DRESULT_INFERENCE = '/home/bugger/Documents/paper/reconstruction/inference'
    DRESULT = '/home/bugger/Documents/paper/reconstruction/results'
    DPLOT = '/home/bugger/Documents/paper/reconstruction/plot'
    print('Not all paths defined')
else:
    DDATA = None
    print('Unknown username ', username)
    sys.exit()

PRETR_SYNTH_APPENDIX = '_PRETR_SYNTH'  # This trains a pretrained Calgary model on synthetic 7T data with RADIAL sampling
PRETR_ACQ_APPENDIX = '_PRETR_ACQ'  # This trains a pretrained Calgary model on 7T data with RADIAL sampling
PRETR_SYNTH_ACQ_APPENDIX = '_PRETR_SYNTH_ACQ'  # This uses a pretrained model on Calgary, finetunes it with synth 7T data, and then with acquisition data.

SCRATCH_SYNTH_APPENDIX = '_SCRATCH_SYNTH'  # This trains a model on synthetic 7T data with RADIAL sampling
SCRATCH_ACQ_APPENDIX = '_SCRATCH_ACQ'  # This trains a model on 7T data with RADIAL sampling
SCRATCH_SYNTH_ACQ_APPENDIX = '_SCRATCH_SYNTH_ACQ'  # This trains a pretrained synth 7T data model on 7T data with RADIAL sampling

# The directories below are not really used anymore...
CIRCUS_APPENDIX = '_CIRCUS'  # This trains a pretrained Calgary model on 7T data with CIRCUS sampling
CIRCUS_SCRATCH_APPENDIX = '_CIRCUS_SCRATCH'  # This trains a model on 7T data with CIRCUS sampling

# List of possible training datasets. Used to create config files
# Should I add 'mixed' to this...?
# I think that messes up the evaluation
ANATOMY_LIST = ['2ch', 'transverse', '4ch', 'sa']
ACCELERATION_LIST = [5, 10]
# Percentage of training data. Used to create config files
PERCENTAGE_LIST = [25, 50, 75, 100]
# The 100, 75, 50 and 25 relate to the PERCENTAGE_LIST
# Based on numeric experiments we see that we have convergence (for the Unet) around 800 iterations.
# We double this to make sure we have really converged and round it up to 2000
# I assume, given the convergence rates I have seen earlier, that this extrapolates to other models
# n_iter_100 = 2000  # This might be enough for Unet.. but not for XPDnet I think
n_iter_100 = 5000
NUM_ITERATIONS_PARALLEL = {100: n_iter_100, 75: 3/4 * n_iter_100, 50: 2/4 * n_iter_100, 25: 1/4 * n_iter_100}
NUM_ITERATIONS_PARALLEL = {k: int(v) for k, v in NUM_ITERATIONS_PARALLEL.items()}
# Seeing the convergence rates, I dont think that this is necessary
# additional_factor = 1.5
# NUM_ITERATIONS_PARALLEL = {k: int(additional_factor * v) for k, v in NUM_ITERATIONS_PARALLEL.items()}

# The names of the metrics as used in the calculate method of class CalculateMetric(FileGatherer)
METRIC_NAMES = ['ssim', 'mse_minmax', 'mse_meanstd', 'flip', 'psnr']
INPUT_METRIC_NAMES = ['ssim_input', 'l2_input', 'psnr_input', 'mse_meanstd_input']
TARGET_METRIC_NAMES = ['ssim_target', 'l2_target', 'psnr_target', 'mse_meanstd_target']
TYPE_NAMES = [PRETR_SYNTH_ACQ_APPENDIX, PRETR_SYNTH_APPENDIX, PRETR_ACQ_APPENDIX,
              SCRATCH_SYNTH_ACQ_APPENDIX, SCRATCH_SYNTH_APPENDIX, SCRATCH_ACQ_APPENDIX]

# These have been selected
# MODEL_NAMES = ["base_conjgradnet", "varnet", "xpdnet", "unet", "kikinet"]
MODEL_NAMES = ["base_conjgradnet", "xpdnet", "unet", "kikinet"]
MODEL_NAME_DICT = {"base_conjgradnet": "Conj. Grad. Net.",
                   "varnet": "Variational Network", "xpdnet": "XPDNet", "unet": "UNet",
                   "kikinet": "KIKI-net"}

METRIC_NAME_DICT = {'ssim': 'SSIM',
                    'l2': 'L2',
                    'mse_minmax': 'MSE',
                    'mse_meanstd': 'MSE',
                    'contrast_ssim': 'SSIM - Contrast',
                    'contrast_glcm': 'GLCM - Contrast',
                    'flip': 'FLIP',
                    'psnr': 'PSNR'}

INPUT_METRIC_NAME_DICT = {k + "_input": v + " input" for k, v in METRIC_NAME_DICT.items()}
TARGET_METRIC_NAME_DICT = {k + "_target": v + " target" for k, v in METRIC_NAME_DICT.items()}

# Old type name dict
# TYPE_NAME_DICT = {'PRETR_ACQ': 'Pretrained + 7T',
#                   'SCRATCH_ACQ': '7T',
#                   'SCRATCH_SYNTH_ACQ': 'Synth 7T + 7T',
#                   'PRETR_SYNTH': 'Pretrained + Synth 7T',
#                   'SCRATCH_SYNTH': 'Synth 7T',
#                   'PRETR_SYNTH_ACQ': 'Pretrained + Synth 7T + 7T'}

# New type name dict
# M1) With pre-trained weights→fine tune on synthetic 7T→fine tune on acquisition 7T
# M2) With pre-trained weights→fine tune on synthetic 7T
# M3) With pre-trained weights→fine tune on acquisition 7T
# M4) Without pre-trained weights→fine tune on synthetic 7T→fine tune on acquisition 7T
# M5) Without pre-trained weights→fine tune on synthetic 7T
# M6) Without pre-trained weights→fine tune on acquisition 7T
TYPE_NAME_DICT = {'SCRATCH_ACQ': 'M6',
                  'SCRATCH_SYNTH': 'M5',
                  'SCRATCH_SYNTH_ACQ': 'M4',
                  'PRETR_ACQ': 'M3',
                  'PRETR_SYNTH': 'M2',
                  'PRETR_SYNTH_ACQ': 'M1'
                  }

# TYPE_ORDERED_LIST = ['SCRATCH_ACQ', 'SCRATCH_SYNTH', 'SCRATCH_SYNTH_ACQ', 'PRETR_ACQ', 'PRETR_SYNTH', 'PRETR_SYNTH_ACQ']

# Make sure that we have both versions of the type name
temp_dict = {"_" + k: v for k, v in TYPE_NAME_DICT.items()}
TYPE_NAME_DICT.update(temp_dict)

TYPE_COLOR_DICT = {'PRETR_ACQ': '#264653',
                  'SCRATCH_ACQ': '#2a9d8f',
                  'SCRATCH_SYNTH_ACQ': '#8ab17d',
                  'PRETR_SYNTH': '#e9c46a',
                  'SCRATCH_SYNTH': '#f4a261',
                  'PRETR_SYNTH_ACQ': '#e76f51'}
temp_dict = {"_" + k: v for k, v in TYPE_COLOR_DICT.items()}
TYPE_COLOR_DICT.update(temp_dict)

TYPE_COLOR_DICT_DARK = {'PRETR_ACQ': '#213D48',
                  'SCRATCH_ACQ': '#24877B',
                  'SCRATCH_SYNTH_ACQ': '#5E8651',
                  'PRETR_SYNTH': '#D19F1F',
                  'SCRATCH_SYNTH': '#EF7C1D',
                  'PRETR_SYNTH_ACQ': '#D7421D'}
temp_dict = {"_" + k: v for k, v in TYPE_COLOR_DICT_DARK.items()}
TYPE_COLOR_DICT_DARK.update(temp_dict)

MODEL_COLOR_DICT = {"base_conjgradnet": '#065143',
              "varnet": '#129490',
              "xpdnet": '#70B77E',
              "unet": "#E0A890",
              "kikinet": "#CE1483"}

MODEL_COLOR_DICT_DARK = {"base_conjgradnet": '#04392F',
                   "varnet": '#0D6D6A',
                   "xpdnet": '#4B955A',
                   "unet": "#BF5F36",
                   "kikinet": "#820D53"}
# These have been randomly selected to be the test cases
# This should be the case in both the u.s. vs .f.s case and the fully sampled test-split of the training data
V_NUMBER_TEST = ['V9_16936', 'V9_17067', 'V9_19531']


# Plot settings
FONTSIZE_XTICKS = FONTSIZE_YTICKS = 28
FONTSIZE_XLABEL = FONTSIZE_YLABEL = 28
FONTSIZE_TITLE = 30
FONTSIZE_LEGEND = 18