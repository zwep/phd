import os
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6

# from multiprocessing import set_start_method
# set_start_method("spawn")
import json
import helper.misc as hmisc
import numpy as np
import re
import objective.inhomog_removal.CalculateMetrics as CalcMetrics
from objective_configuration.inhomog_removal import get_path_dict, MODEL_DIR, LOG_DIR, IMG_SYNTH_COIL
import argparse
from loguru import logger
import sys


# These functions below are mostly used only to reduce the amount of code in the script....
def _dict2list(listdict, filelist, slicelist):
    dict_list = hmisc.listdict2dictlist(listdict)
    dict_list['file_list'] = filelist
    dict_list['slice_list'] = slicelist
    return dict_list


def _store_container_item(container_item, file_list, slice_list, dest_dir, dest_name):
    # Very special container. Containig a lot of stuff
    container_dict = _dict2list(container_item, file_list, slice_list)
    temp_serialized = json.dumps(container_dict)
    with open(os.path.join(dest_dir, dest_name), 'w') as f:
        f.write(temp_serialized)


try:
    # Check if we have a __file__ name
    file_base_name = hmisc.get_base_name(__file__)
    logger.add(os.path.join(LOG_DIR, f"{file_base_name}.log"))
except NameError:
    print('No file name known. Not reporting to logger.')


parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='Provide the name of the directory that we want to post process')
parser.add_argument('-dataset', type=str, default='all',
                    help='Provide the name of the dataset on which we want to evaluate: '
                         'synthetic, 3T, patient, volunteer')

parser.add_argument('--mid_slice', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')

parser.add_argument('--calc_pred_metric', default=True, action='store_true')
parser.add_argument('--no-calc_pred_metric', dest='calc_pred_metric', action='store_false')

parser.add_argument('--calc_target_metric', default=True, action='store_true')
parser.add_argument('--no-calc_target_metric', dest='calc_target_metric', action='store_false')

parser.add_argument('--calc_input_metric', default=True, action='store_true')
parser.add_argument('--no-calc_input_metric', dest='calc_input_metric', action='store_false')


p_args = parser.parse_args()
path = p_args.path
dataset = p_args.dataset
debug = p_args.debug
mid_slice_option = p_args.mid_slice
calculate_pred_metric = p_args.calc_pred_metric
calculate_target_metric = p_args.calc_target_metric
calculate_input_metric = p_args.calc_input_metric

print('Feedback on argparse')
print(f'Calculate prediction metric: {calculate_pred_metric}')
print(f'Calculate target metric: {calculate_target_metric}')
print(f'Calculate input metric: {calculate_input_metric}')
print(f'Calculate only mid slice: {mid_slice_option}')



# Get all paths..
dconfig = os.path.join(MODEL_DIR, path)
path_dict, dataset_list = get_path_dict(dconfig)

# Check what kind of synthetic data input we need (multi coil or single)
single_input = True
if 'multi' in path:
    single_input = False
    # Use the individual coil images path as input
    # Why..?
    # path_dict['synthetic']['dimage'] = IMG_SYNTH_COIL
    # Limit the number of datasets we check
    dataset_list = ['volunteer', 'synthetic']


if dataset == 'all':
    sel_dataset_list = dataset_list
else:
    if dataset in dataset_list:
        sel_dataset_list = [dataset]
    else:
        logger.debug(f'Unknown dataset selected: {dataset}')
        sys.exit()


for dataset_name in sel_dataset_list:
    temp_dict = path_dict[dataset_name]
    temp_ddest = temp_dict['dpred']
    # Hacking it like this....
    # Who wont these things work....
    # Why do I re-define this one? I think that is uncessesary
    # I think these are the storage locations, not where we get hte data...
    # temp_dict['dimage'] = os.path.join(temp_ddest, 'input')
    # temp_dict['dmask'] = os.path.join(temp_ddest, 'mask')
    temp_dict['dpred'] = os.path.join(temp_ddest, 'pred')
    # temp_dict['dtarget'] = os.path.join(temp_ddest, 'target')
    # Options for testing stuff...
    # Aha! so this might be causing some issues...
    temp_dict['mid_slice'] = mid_slice_option
    temp_dict['debug'] = debug
    temp_dict['shrink_pixels'] = 30
    logger.info(f"Input path {temp_dict['dimage']}")
    mask_dir_name = os.path.basename(temp_dict['dmask'])
    mask_name = re.sub('_mask', '', mask_dir_name)
    dest_dir = os.path.dirname(temp_dict['dpred'])
#
    if dataset_name == 'volunteer':
        # Needed for volunteer 7T data
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=10*10, **temp_dict)
        # metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.nii.gz', patch_size=10 * 10, **temp_dict)
        metric_obj.glcm_dist = list(range(10))[1:]
    elif dataset_name == 'patient_3T':
        # Needed for patient 3T
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.h5', patch_size=7*10, mask_suffix='_target', **temp_dict)
        # metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.nii.gz', patch_size=7 * 10, **temp_dict)
        metric_obj.glcm_dist = list(range(7))[1:]
    elif dataset_name == 'patient':
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.npy', patch_size=16*10, **temp_dict)
        # metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.nii.gz', patch_size=16 * 10, **temp_dict)
        # We want 5mm and we have a pixel spacing of approx 0.28mm
        metric_obj.glcm_dist = list(range(16))[1:]
    elif dataset_name == 'synthetic':
        # In the test set we have a pixel spacing of...
        metric_obj = CalcMetrics.CalculateMetrics(mask_ext='.nii.gz', patch_size=7*10, **temp_dict)
        # We want 5mm and we have a pixel spacing of approx 0.7mm
        metric_obj.glcm_dist = list(range(7))[1:]
    else:
        logger.info(f'Unknown dataset {dataset_name}. Continueing with loop')
        continue
#
    if debug:
        print(" ONLY USING THREE FILES TO MAKE SURE THAT WE CAN QUICKLY CHECK SOME RESULTS")
        metric_obj.file_list = metric_obj.file_list[0:3]
#
    if calculate_pred_metric:
        glcm_rel, glcm_input, glcm_pred, coef_var_rel, coef_var_input, coef_var_pred, slice_list = metric_obj.run_features()
        _store_container_item(glcm_rel, dest_name=f'{mask_name}_rel_change_glcm.json',
                              file_list=metric_obj.file_list, slice_list=slice_list, dest_dir=dest_dir)
        _store_container_item(glcm_input, dest_name=f'{mask_name}_input_change_glcm.json',
                              file_list=metric_obj.file_list, slice_list=slice_list, dest_dir=dest_dir)
        _store_container_item(glcm_pred, dest_name=f'{mask_name}_pred_change_glcm.json',
                              file_list=metric_obj.file_list, slice_list=slice_list, dest_dir=dest_dir)
        np.save(os.path.join(dest_dir, f'{mask_name}_rel_coef_of_variation.npy'), coef_var_rel)
        np.save(os.path.join(dest_dir, f'{mask_name}_input_coef_of_variation.npy'), coef_var_input)
        np.save(os.path.join(dest_dir, f'{mask_name}_pred_coef_of_variation.npy'), coef_var_pred)
#
    if calculate_target_metric and (metric_obj.dtarget is not None):
        coefv_target_rel, coefv_target, glcm_rel, glcm_target, RMSE_list, SSIM_list, WSS_distance, slice_list = metric_obj.run_features_target()
        _store_container_item(glcm_target, dest_name=f'{mask_name}_target_change_glcm.json',
                              file_list=metric_obj.file_list, slice_list=slice_list, dest_dir=dest_dir)
        _store_container_item(glcm_rel, dest_name=f'{mask_name}_rel_target_change_glcm.json',
                              file_list=metric_obj.file_list, slice_list=slice_list, dest_dir=dest_dir)
        np.save(os.path.join(dest_dir, f'{mask_name}_rel_target_coef_of_variation.npy'), coefv_target_rel)
        np.save(os.path.join(dest_dir, f'{mask_name}_target_coef_of_variation.npy'), coefv_target)
        np.save(os.path.join(dest_dir, f'{mask_name}_rmse_values.npy'), RMSE_list)
        np.save(os.path.join(dest_dir, f'{mask_name}_ssim_values.npy'), SSIM_list)
        np.save(os.path.join(dest_dir, f'{mask_name}_wasserstein_values.npy'), WSS_distance)

    if calculate_input_metric and (metric_obj.dtarget is not None):
        RMSE_list, SSIM_list, WSS_distance, slice_list = metric_obj.run_target_input_features()
        np.save(os.path.join(dest_dir, f'{mask_name}_input_rmse.npy'), RMSE_list)
        np.save(os.path.join(dest_dir, f'{mask_name}_input_ssim.npy'), SSIM_list)
        np.save(os.path.join(dest_dir, f'{mask_name}_input_wasserstein.npy'), WSS_distance)

# Nice read..? For later...
# https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0212110&type=printable
# Gray-level invariant Haralick texture features
