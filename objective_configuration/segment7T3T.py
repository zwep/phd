import os
import sys
import re
from PIL import ImageColor
from matplotlib.colors import ListedColormap
import numpy as np

from nnunet.paths import nnUNet_raw_data

# Check whether we are working local or not..
username = os.environ.get('USER', os.environ.get('USERNAME'))

# HPC
if username == '20184098':
    DCODE = '/home/bme001/20184098/code/pytorch_in_mri'
    DFINAL = '/home/bme001/20184098/paper/segment7T3T'
    DCMR7T = '/home/bme001/20184098/data/cmr7t3t'
    DCODE_GAN = os.path.join(DCMR7T, 'code/CMR_CUT_7T_Seb')
    DLOG = '/home/bme001/20184098/project/segment7T3T'
# Legolas/Boromir...
elif username == 'seb':
    DCODE = '/data/seb/code/pytorch_in_mri'
    DCODE_GAN = '/data/cmr7t3t/code/CMR_CUT_7T_Seb'
    dankebrand_pkl = '/data/seb/resnet34_5percent_size256_extremeTfms_ceLoss_fastai2.pkl'
    DFINAL = '/data/seb/paper/segment7T3T'
    DLOG = '/data/seb/project/segment7T3T'
# UMC
elif username == 'sharreve':
    DCODE = '/home/sharreve/local_scratch/code/pytorch_in_mri'
    DCMR7T = '/home/sharreve/local_scratch/cmr7t3t'
    # DCODE_GAN = '/data/cmr7t3t/code/CMR_CUT_7T_Seb'
    # dankebrand_pkl = '/data/seb/resnet34_5percent_size256_extremeTfms_ceLoss_fastai2.pkl'
    DFINAL = '/home/sharreve/local_scratch/paper/segment7T3T'
    DLOG = '/home/sharreve/local_scratch/project/segment7T3T'
elif username == 'bugger':
    DCODE = '/home/bugger/PycharmProjects/pytorch_in_mri'
    dankebrand_pkl = '/home/bugger/Documents/paper/segment7t3t/alternative_model/resnet34_5percent_size256_extremeTfms_ceLoss_fastai2.pkl'
    DCODE_GAN = '/home/bugger/PycharmProjects/CMR_CUT_7T_Seb'
    DFINAL = '/home/bugger/paper/segment7T3T'
    DLOG = '/home/bugger/Documents/paper/segment7T3T'
else:
    print('Unknown username ', username)
    sys.exit()

DATASET_LIST = ['7t', 'acdc', 'mm1a', 'mm1b', 'kaggle', 'mm2']
DATASET_SYNTH_LIST = ['acdc_gan', 'acdc_5poly', 'acdc_biasfield', 'mm1a_gan', 'mm1a_biasfield', 'mm1a_5poly',
                      'mm1b_gan', 'mm1b_biasfield', 'mm1b_5poly']


def get_path_dict(dataset):
    dataset = dataset.lower()
    # needed to avoid  undefined variables..
    ddata_img = None
    ddata_label = None
    ddata_model_results = None
    ddest_PNG = None

    if dataset in ['7t']:
        # ddata_img = '/data/cmr7t3t/cmr7t/Image_ED_ES'
        # ddata_label = '/data/cmr7t3t/cmr7t/Label_ED_ES'
        ddata_img = os.path.join(nnUNet_raw_data, 'Task999_7T/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task999_7T/labelsTs')
        ddata_model_results = os.path.join(DCMR7T, 'cmr7t/Results')
        ddest_PNG = os.path.join(DCMR7T, 'cmr7t/Results_PNG')
    elif dataset in ['acdc']:
        ddata_img = os.path.join(nnUNet_raw_data, 'Task511_ACDC/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task511_ACDC/labelsTs')
        ddata_model_results = os.path.join(DCMR7T, 'acdc/acdc_processed/Results')
        ddest_PNG = os.path.join(DCMR7T, 'acdc/acdc_processed/Results_PNG')
    elif dataset in ['acdc_biasfield']:
        # Only used to check the features of this...
        # imageTr is needed here, since Ts is empty
        ddata_img = os.path.join(nnUNet_raw_data, 'Task611_Biasfield_ACDC/imagesTr')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task611_Biasfield_ACDC/labelsTr')
    elif dataset in ['acdc_gan']:
        # Only used to check the features of this...
        ddata_img = os.path.join(nnUNet_raw_data, 'Task610_GAN_ACDC/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task610_GAN_ACDC/labelsTs')
    elif dataset in ['acdc_5poly']:
        # Only used to check the features of this...
        # imageTr is needed here, since Ts is empty
        ddata_img = os.path.join(nnUNet_raw_data, 'Task613_ACDC/imagesTr')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task613_ACDC/labelsTr')
    elif dataset in ['mm1a']:
        ddata_img = os.path.join(nnUNet_raw_data, 'Task501_MM1_A/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task501_MM1_A/labelsTs')
        ddata_model_results = os.path.join(DCMR7T, 'mms1/all_phases_mid/Vendor_A/Results')
        ddest_PNG = os.path.join(DCMR7T, 'mms1/all_phases_mid/Vendor_A/Results_PNG')
    elif dataset in ['mm1a_biasfield']:
        # Only used to check the features of this...
        ddata_img = os.path.join(nnUNet_raw_data, 'Task901_Biasfield_MM1_A_balanced/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task901_Biasfield_MM1_A_balanced/labelsTs')
    elif dataset in ['mm1a_5poly']:
        # Only used to check the features of this...
        ddata_img = os.path.join(nnUNet_raw_data, 'Task905_Biasfield_MM1_A_balanced/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task905_Biasfield_MM1_A_balanced/labelsTs')
    elif dataset in ['mm1a_gan']:
        # Only used to check the features of this...
        ddata_img = os.path.join(nnUNet_raw_data, 'Task903_GAN_MM1_A_balanced/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task903_GAN_MM1_A_balanced/labelsTs')
    elif dataset in ['mm1b']:
        ddata_img = os.path.join(nnUNet_raw_data, 'Task502_MM1_B/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task502_MM1_B/labelsTs')
        ddata_model_results = os.path.join(DCMR7T, 'mms1/all_phases_mid/Vendor_B/Results')
        ddest_PNG = os.path.join(DCMR7T, 'mms1/all_phases_mid/Vendor_B/Results_PNG')
    elif dataset in ['mm1b_gan']:
        # Only used to check the features of this...
        ddata_img = os.path.join(nnUNet_raw_data, 'Task904_GAN_MM1_B_balanced/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task904_GAN_MM1_B_balanced/labelsTs')
    elif dataset in ['mm1b_biasfield']:
        # Only used to check the features of this...
        ddata_img = os.path.join(nnUNet_raw_data, 'Task902_Biasfield_MM1_B_balanced/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task902_Biasfield_MM1_B_balanced/labelsTs')
    elif dataset in ['mm1b_5poly']:
        # Only used to check the features of this...
        ddata_img = os.path.join(nnUNet_raw_data, 'Task906_Biasfield_MM1_B_balanced/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task906_Biasfield_MM1_B_balanced/labelsTs')
    elif dataset in ['kaggle', 'anken']:
        ddata_img = os.path.join(nnUNet_raw_data, 'Task998_ankenbrand/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task998_ankenbrand/labelsTs')
        ddata_model_results = os.path.join(DCMR7T, 'ankenbrand_data/Results')
        ddest_PNG = os.path.join(DCMR7T, 'ankenbrand_data/Results_PNG')
    elif dataset in ['mm2']:
        ddata_img = os.path.join(nnUNet_raw_data, 'Task997_mm2/imagesTs')
        ddata_label = os.path.join(nnUNet_raw_data, 'Task997_mm2/labelsTs')
        ddata_model_results = os.path.join(DCMR7T, 'mms2/Results')
        ddest_PNG = os.path.join(DCMR7T, 'mms2/Results_PNG')
    else:
        print("Please set the options -dataset and -model")
        print("Options for -dataset: 7T, ACDC, mm1a, mm1b")
        print("Options for -model: 0,1,2,3,4,...")
        sys.exit()
    name_dice_scores = 'model_dice_score.json'
    name_hausdorf_scores = 'model_hausdorf_score.json'
    name_jaccard_scores = 'model_jaccard_score.json'
    name_assd_scores = 'model_assd_score.json'

    if ddata_model_results is None:
        path_dict = {'dimg': ddata_img, 'dlabel': ddata_label,
                     'dresults': ddata_model_results,'dpng': ddest_PNG}
    else:
        ddata_dice = os.path.join(ddata_model_results, name_dice_scores)
        ddata_hausdorf = os.path.join(ddata_model_results, name_hausdorf_scores)
        ddata_jaccard = os.path.join(ddata_model_results, name_jaccard_scores)
        ddata_assd = os.path.join(ddata_model_results, name_assd_scores)
        path_dict = {'dimg': ddata_img, 'dlabel': ddata_label, 'dresults': ddata_model_results,
                     'dpng': ddest_PNG, 'ddice': ddata_dice, 'dhausdorf': ddata_hausdorf,
                     'djaccard': ddata_jaccard, 'dassd': ddata_assd}
    return path_dict


CLASS_INTERPRETATION = {'1': 'RV', '2': 'MYO', '3': 'LV'}
COLOR_DICT = {'1': '#1E88E5', '2': '#D81B60', '3': '#004D40'}
COLOR_DICT_RGB = {k: np.array(ImageColor.getcolor(x, "RGB"))/256 for k, x in COLOR_DICT.items()}
CMAP_ALL = np.array(list(COLOR_DICT_RGB.values()))
MY_CMAP = ListedColormap(CMAP_ALL)

DICT_DATASET_DLABEL = {i_dataset: get_path_dict(i_dataset)['dlabel'] for i_dataset in DATASET_LIST + DATASET_SYNTH_LIST}
DICT_DATASET_DIMG = {i_dataset: get_path_dict(i_dataset)['dimg'] for i_dataset in DATASET_LIST + DATASET_SYNTH_LIST}

TRANSFORM_MODEL_NAMES = \
    {
     '501': 'M&Ms 1 A',  # M&Ms 1 A
     '502': 'M&Ms 1 B',  # M&Ms 1 B
     '511': 'ACDC',  # ACDC
     '513': 'Kaggle',  # Kaggle
     '514': 'M&Ms 2',  # M&Ms 2
     'synth-model-31-03': 'GAN based',  # ACDC
     '610': 'GAN based',  # ACDC
     '611': 'Simulated bias field',  # ACDC
     '613': '5-Polynomial bias field',  # ACDC
     '614': '2-Polynomial bias field',  # ACDC
     '630': 'M&Ms 1 A, B, ACDC',
     '631': 'Simulated bias field (large)',
     '633': 'Polynomial bias field (large)',
     '635': 'GAN based (large)',
     '901': 'Simulated bias field',  # MM 1 A
     '902': 'Simulated bias field',  # MM 1 B
     '903': 'GAN based',  # MM 1 A
     '904': 'GAN based',  # MM 1 B
     '905': '5-Polynomial bias field',  # MM 1 A
     '906': '5-Polynomial bias field',  # MM 1 B
     '907': '50% Poly 50% Simu bias field ',  # MM 1 A
     '908': '50% Poly 50% Simu bias field ',  # MM 1 B
     '909': '2-Polynomial bias field',  # MM 1 A
     '910': '2-Polynomial bias field',  # MM 1 B
     'ground_truth': 'Ground truth'
     }


raw_listdir = os.listdir(nnUNet_raw_data)
TASK_NR_TO_DIR = {}
for i_dir in raw_listdir:
    if re.findall("Task([0-9]{3})", i_dir):
        temp = {re.findall("Task([0-9]{3})", i_dir)[0]: i_dir}
        TASK_NR_TO_DIR.update(temp)

TASK_NR_TO_DIR['ground_truth'] = 'ground_truth'

DIR_TO_TASK_NR = {v: k for k, v in TASK_NR_TO_DIR.items()}


