import os
import time
import pandas as pd
import helper.misc as hmisc
from objective_configuration.reconstruction import DMODEL, DRESULT, DRESULT_INFERENCE, PERCENTAGE_LIST, TYPE_NAMES, MODEL_NAMES
import glob

"""

I am tired of constantly checking folders..

So we want to check is model trained and go through the %p trained folders
There we will encouter the `train_mixed` folder.

In that folder: check for .pt files and extract the latest one. Present that.
Store found information in a pandas dataframe and save   
"""


def find_first_last_modified_file(folder, ext='.pt', ind=0):
    # Check for .pt files in the folder by default
    file_list = glob.glob(os.path.join(folder, f"*{ext}"))
    if file_list:
        # Sort the file list based on modification time (newest first)
        sorted_files = sorted(file_list, key=os.path.getmtime, reverse=True)
        sel_file = sorted_files[ind]
        return sel_file, time.ctime(os.path.getmtime(sel_file))
    return None, None


df = pd.DataFrame(columns=["Model", "Type", "Percentage", "Latest Model", "Model time",
                           "First result file", "Last result file",
                           "First inference file", "Last inference file", "Metric result", "Metric inference"])

for i_model in MODEL_NAMES:
    for i_type in TYPE_NAMES:
        model_type = f'{i_model}{i_type}'
        # Search for model files
        model_path = os.path.join(DMODEL, model_type)
        model_result_path = os.path.join(DRESULT, model_type)
        model_inference_path = os.path.join(DRESULT_INFERENCE, model_type)
        for i_percentage in [0] + PERCENTAGE_LIST:
            if i_percentage == 0:
                inference_str_suffix = 'pretrained/undersampled'
                result_str_suffix = 'pretrained/mixed/5x'
            else:
                inference_str_suffix = 'train_mixed/undersampled'
                result_str_suffix = 'train_mixed/mixed/5x'

            percentage_model_path = os.path.join(model_path, str(i_percentage) + 'p', 'train_mixed')
            result_path = os.path.join(model_result_path, str(i_percentage) + 'p', result_str_suffix)
            inference_path = os.path.join(model_inference_path, str(i_percentage) + 'p', inference_str_suffix)

            print(f'Result path {result_path}')
            _, last_result_time = find_first_last_modified_file(result_path, ext='.h5', ind=0)
            _, first_result_time = find_first_last_modified_file(result_path, ext='.h5', ind=-1)

            print(f'Inference path {inference_path}')
            _, last_inference_time = find_first_last_modified_file(inference_path, ext='.h5', ind=0)
            _, first_inference_time = find_first_last_modified_file(inference_path, ext='.h5', ind=-1)

            _, json_inference_time = find_first_last_modified_file(model_inference_path, ext='.json', ind=-1)
            _, json_result_time = find_first_last_modified_file(model_result_path, ext='.json', ind=-1)

            latest_pt_file, pt_time = find_first_last_modified_file(percentage_model_path, ext='.pt', ind=0)
            if latest_pt_file:
                latest_pt_file_name = os.path.basename(latest_pt_file)
            else:
                latest_pt_file_name = latest_pt_file

            df = df.append({"Model": i_model, "Type": i_type, "Percentage": i_percentage,
                            "Latest Model": latest_pt_file_name, "Model time": pt_time,
                            "First result file": first_result_time, "Last result file": last_result_time,
                            "First inference file": first_inference_time, "Last inference file": last_inference_time,
                            "Metric inference": json_inference_time,
                            "Metric result": json_result_time},
                           ignore_index=True)


df.to_csv(os.path.join(DMODEL, "model_information.csv"), index=False)
n_divider = 30
print('Model', ' ' * (n_divider - 5), 'Percentage', ' ' * (n_divider - 10), 'Latest Model', ' ' * (n_divider - 12))
prev_model_name = ''
for i, irow in df.iterrows():
    model_name = irow['Model']
    if prev_model_name != model_name:
        print('\n')
    n_model = len(model_name)
    perc_value = irow['Percentage']
    n_perc = len(str(perc_value))
    latest_name = irow['Latest Model']
    if latest_name is None:
        latest_name = 'None'
    n_latest = len(latest_name)
    print(model_name, ' ' * (n_divider - n_model),
          perc_value, ' ' * (n_divider - n_perc),
          latest_name, ' ' * (n_divider - n_latest))
    prev_model_name = model_name