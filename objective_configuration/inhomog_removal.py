import os
import sys
# Check whether we are working local or not..
username = os.environ.get('USER', os.environ.get('USERNAME'))

# UMC server
if username == 'sharreve':
    base_dir = '/home/sharreve/local_scratch'
    data_dir = os.path.join(base_dir, 'mri_data')
elif username == 'bugger':
    # This is not correct... but approximately it will be
    base_dir = '/home/bugger/Documents'
    data_dir = os.path.join(base_dir, 'data')
else:
    print('Unknown username ', username)
    sys.exit()


def get_path_dict(model_config_path):
    PATH_DICT = {'volunteer':
                         {'dimage': IMG_VOLUNTEER, 'dmask': MASK_VOLUNTEER,
                          'dtarget': None,
                          'dpred': os.path.join(model_config_path, 'volunteer_corrected')},
                 'patient_3T':
                         {'dimage': IMG_3T, 'dmask': MASK_3T,
                          'dtarget': None,
                          'dpred': os.path.join(model_config_path, "patient_corrected_3T")},
                 'patient':
                         {'dimage': IMG_PATIENT, 'dmask': MASK_PATIENT,
                          'dtarget': None,
                          'dpred': os.path.join(model_config_path, "patient_corrected")},
                 'synthetic':
                         {'dimage': IMG_SYNTH, 'dmask': MASK_SYNTH,
                          'dtarget': TARGET_SYNTH,
                          "dpred": os.path.join(model_config_path, "target_corrected")}
                 }
    # Now I also need to define N4ITK stuff
    DATASET_LIST = list(PATH_DICT.keys())
    return PATH_DICT, DATASET_LIST


PLOT_DIR = os.path.join(base_dir, 'paper/inhomog_removal/plots')
MODEL_DIR = os.path.join(base_dir, 'model_run')
LOG_DIR = os.path.join(base_dir, 'paper/inhomog_removal/log')

IMG_VOLUNTEER = os.path.join(data_dir, 'volunteer_data/t2w_nifti')
MASK_VOLUNTEER = os.path.join(data_dir, 'volunteer_data/body_mask')

IMG_SYNTH = os.path.join(data_dir, 'registrated_h5/test_nifti/input_abs_sum_nifti')
IMG_SYNTH_COIL = os.path.join(data_dir, 'registrated_h5/test_nifti/input_nifti')
MASK_SYNTH = os.path.join(data_dir, 'registrated_h5/test_nifti/mask_b1')
TARGET_SYNTH = os.path.join(data_dir, 'registrated_h5/test_nifti/target')

IMG_PATIENT = os.path.join(data_dir, 'daan_reesink/image_nifti')
MASK_PATIENT = os.path.join(data_dir, 'daan_reesink/mask')

IMG_3T = os.path.join(data_dir, 'prostate_weighting_h5/test/target_nifti')
MASK_3T = os.path.join(data_dir, 'prostate_weighting_h5/test/mask')

INHOMONET_PATH = os.path.join(MODEL_DIR, 'selected_inhomog_removal_models/inhomonet')
INHOMONET_WEIGHTS = os.path.join(data_dir, 'pretrained_networks/inhomonet')

CHOSEN_FEATURE = ['homogeneity', 'energy']

"""
Create dirs if not available
"""

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)




# base_pred = '/local_scratch/sharreve/model_run/selected_inhomog_removal_models'

# Not sure yet what to do with this....
# Remember this.. but using a different approach now.
# Using the following now:
#   dconfig = os.path.join(MODEL_DIR, path)
#   path_dict, dataset_list = get_path_dict(dconfig)
# RESULT_PATH = {"single_biasf_patient": os.path.join(base_pred, 'single_biasfield/patient_corrected'),
#             "single_homog_patient": os.path.join(base_pred, 'single_homogeneous/patient_corrected'),
#             "single_biasf_patient_3T": os.path.join(base_pred, 'single_biasfield/patient_corrected_3T'),
#             "single_homog_patient_3T": os.path.join(base_pred, 'single_homogeneous/patient_corrected_3T'),
#             "multi_biasf_volunteer": os.path.join(base_pred, 'multi_biasfield/volunteer_corrected'),
#             "multi_homog_volunteer": os.path.join(base_pred, 'multi_homogeneous/volunteer_corrected'),
#             "single_biasf_volunteer": os.path.join(base_pred, 'single_biasfield/volunteer_corrected'),
#             "single_homog_volunteer": os.path.join(base_pred, 'single_homogeneous/volunteer_corrected'),
#             "multi_biasf_test": os.path.join(base_pred, 'multi_biasfield/target_corrected'),
#             "multi_homog_test": os.path.join(base_pred, 'multi_homogeneous/target_corrected'),
#             "single_biasf_test": os.path.join(base_pred, 'single_biasfield/target_corrected'),
#             "single_homog_test": os.path.join(base_pred, 'single_homogeneous/target_corrected'),
#             "n4_patient": "/local_scratch/sharreve/mri_data/daan_reesink/image_n4itk",
#             "n4_volunteer": "/local_scratch/sharreve/mri_data/volunteer_data/t2w_n4itk",
#             "n4_patient_3T": "/local_scratch/sharreve/mri_data/prostate_weighting_h5/test",
#             "n4_test": "/local_scratch/sharreve/mri_data/registrated_h5/test_nifti"}

# ddata_volunteer_n4 = '/local_scratch/sharreve/mri_data/volunteer_data/t2w_n4itk'
# ddata_patient_n4 = '/local_scratch/sharreve/mri_data/daan_reesink/image_n4itk'
# ddata_test_n4 = '/local_scratch/sharreve/mri_data/registrated_h5/test_nifti'