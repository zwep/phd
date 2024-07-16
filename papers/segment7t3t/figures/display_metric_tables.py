import helper.array_transf as harray
import os
import pandas as pd
import numpy as np
import helper.misc as hmisc
from objective_configuration.segment7T3T import get_path_dict, DATASET_LIST, TRANSFORM_MODEL_NAMES, DIR_TO_TASK_NR
import sys
import argparse


def transform_names(x):
    # Very funny way to translate a directory name to our in-paper abbreviation
    # This function is used to create better table names
    if x.startswith('Task'):
        new_name = TRANSFORM_MODEL_NAMES[DIR_TO_TASK_NR[x]]
    else:
        new_name = TRANSFORM_MODEL_NAMES[x]
    return new_name


def json_to_dataframe_str_mean_std(json_temp, col_name, new_column_name, sel_model_list, nround=2):
    # Here we can easily transform a json file to a dataframe and print its mean/std in a string like fashion
    df_temp = harray.nested_dict_to_df(json_temp, column_name=col_name)
    df_temp = df_temp.rename_axis(["Model", "Subject", "Class", "Phase"])
    available_index = [x for x in sel_model_list if x in df_temp.index]
    not_available_index = [x for x in sel_model_list if not x in df_temp.index]
    df_temp = df_temp.loc[available_index]
    df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_temp.dropna(how="all", inplace=True)
    aggr_model = df_temp.groupby(by=["Model"]).mean().round(nround).astype(str).apply(lambda x: x.str.ljust(nround+2, '0'))
    std_model = df_temp.groupby(by=["Model"]).std().round(nround).astype(str).apply(lambda x: x.str.ljust(nround+2, '0'))
    df_str = aggr_model + " ± " + std_model
    df_str.columns = [new_column_name]
    df_str.rename(index=transform_names, inplace=True)
    if len(not_available_index):
        print("The following model indices were not found: ")
        [print(f'\t {x}') for x in not_available_index]
    return df_str


# json_temp = dice_json
# col_name = 'dice',
# new_column_name = new_column_name
# sel_model_list = sel_model_name_list

def json_to_dataframe_str_median_iqr_model_class(json_temp, col_name, new_column_name, sel_model_list, nround=2):
    # Here we can easily transform a json file to a dataframe and print its mean/std in a string like fashion
    df_temp = harray.nested_dict_to_df(json_temp, column_name=col_name)
    df_temp = df_temp.rename_axis(["Model", "Subject", "Class", "Phase"])
    available_index = [x for x in sel_model_list if x in df_temp.index]
    not_available_index = [x for x in sel_model_list if not x in df_temp.index]
    df_temp = df_temp.loc[available_index]
    df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_temp.dropna(how="all", inplace=True)
    aggr_model = df_temp.groupby(by=["Model", "Class"]).median().round(nround).astype(str).apply(lambda x: x.str.ljust(nround+2, '0'))
    Q1_model = df_temp.groupby(by=["Model", "Class"]).quantile(0.25).round(nround)
    Q3_model = df_temp.groupby(by=["Model", "Class"]).quantile(0.75).round(nround)
    IQR_model = (Q3_model - Q1_model).round(nround).astype(str).apply(lambda x: x.str.ljust(nround + 2, '0'))
    df_str = aggr_model + " (" + IQR_model + ")"
    df_str.columns = [new_column_name]
    if len(not_available_index):
        print("The following model indices were not found: ")
        [print(f'\t {x}') for x in not_available_index]
    return df_str


def json_to_dataframe_str_median_iqr_model(json_temp, col_name, new_column_name, sel_model_list, nround=2):
    # Here we can easily transform a json file to a dataframe and print its mean/std in a string like fashion
    df_temp = harray.nested_dict_to_df(json_temp, column_name=col_name)
    df_temp = df_temp.rename_axis(["Model", "Subject", "Class", "Phase"])
    available_index = [x for x in sel_model_list if x in df_temp.index]
    not_available_index = [x for x in sel_model_list if not x in df_temp.index]
    df_temp = df_temp.loc[available_index]
    df_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
    print('Number of NA before the drop', sel_model_list, df_temp.isna().sum().values)
    df_temp.dropna(how="all", inplace=True)
    aggr_model = df_temp.groupby(by=["Model"]).median().round(nround).astype(str).apply(lambda x: x.str.ljust(nround+2, '0'))
    Q1_model = df_temp.groupby(by=["Model"]).quantile(0.25).round(nround)
    Q3_model = df_temp.groupby(by=["Model"]).quantile(0.75).round(nround)
    IQR_model = (Q3_model - Q1_model).round(nround).astype(str).apply(lambda x: x.str.ljust(nround + 2, '0'))
    df_str = aggr_model + " (" + IQR_model + ")"
    df_str.columns = [new_column_name]
    df_str.rename(index=transform_names, inplace=True)
    if len(not_available_index):
        print("The following model indices were not found: ")
        [print(f'\t {x}') for x in not_available_index]
    return df_str


"""
I want to display all models and their metrics...
"""

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default=None)
parser.add_argument('-sd', type=str, default=None)
parser.add_argument('-fun', type=str, default=None)
p_args = parser.parse_args()
model_selection = p_args.model
source_data = p_args.sd  # Should be something like ACDC, MM1A, MM1B...
conversion_function_name = p_args.fun

# model_selection = 't511'
# source_data = 'acdc'
# conversion_function_name = 'iqr'

select_models = True
if conversion_function_name in ['iqr', 'median', 'model']:
    json_to_str_fun = json_to_dataframe_str_median_iqr_model
elif conversion_function_name in ['class']:
    select_models = False
    json_to_str_fun = json_to_dataframe_str_median_iqr_model_class
else:
    print('Unknown option for conversion ', conversion_function_name)
    sys.exit()


all_result_dir = []
for i_dataset in DATASET_LIST:
    path_dict = get_path_dict(i_dataset)
    all_result_dir.append(path_dict['dresults'])

all_model_result_dir = []
for i_result_dir in all_result_dir:
    model_name_list = [x for x in os.listdir(i_result_dir) if
                       os.path.isdir(os.path.join(i_result_dir, x))]
    all_model_result_dir.extend(model_name_list)

all_model_result_dir = sorted(list(set(all_model_result_dir)))

print("List of model names:")
for i, imodelname in enumerate(all_model_result_dir):
    print(i, '\t', imodelname)

import objective_helper.segment7T3T as hsegm7t
if model_selection:
    sel_model_name_list = hsegm7t.model_selection_processor(model_selection, all_model_result_dir)
else:
    print("Please select a model first..")
    sys.exit()


"""
Now that we have the models we would like... display all..
"""

overal_dice_table = []
overal_hausdorff_table = []
for i_dataset in DATASET_LIST:
    print(i_dataset)
    path_dict = get_path_dict(i_dataset)
    # Load hausdorf and dice
    if os.path.isfile(path_dict['ddice']):
        dice_json = hmisc.load_json(path_dict['ddice'])
        new_column_name = f'dice {i_dataset}'
        dataframe_str = json_to_str_fun(dice_json, 'dice', new_column_name, sel_model_name_list)
        overal_dice_table.append(dataframe_str)
    if os.path.isfile(path_dict['dhausdorf']):
        hausdorf_json = hmisc.load_json(path_dict['dhausdorf'])
        new_column_name = f'hausdorf {i_dataset}'
        dataframe_str = json_to_str_fun(hausdorf_json, 'hausdorf', new_column_name, sel_model_name_list, nround=1)
        overal_hausdorff_table.append(dataframe_str)


sel_col_dice = [f'dice {source_data.lower()}', 'dice 7t', 'dice mm2', 'dice kaggle']
sel_col_hd = [f'hausdorf {source_data.lower()}', 'hausdorf 7t', 'hausdorf mm2', 'hausdorf kaggle']
overal_dice_df = pd.concat(overal_dice_table, axis=1)
overal_hausdorff_df = pd.concat(overal_hausdorff_table, axis=1)
print(f'Dice score from {source_data}')
if select_models:
    sel_col_dice = [x for x in overal_dice_df.columns if x in sel_col_dice]
    print(overal_dice_df[sel_col_dice].style.to_latex())
else:
    print(overal_dice_df.style.to_latex())

print(f'Hausdorff distance from {source_data}')
if select_models:
    sel_col_hd = [x for x in overal_hausdorff_df.columns if x in sel_col_hd]
    print(overal_hausdorff_df[sel_col_hd].style.to_latex())
else:
    print(overal_hausdorff_df.style.to_latex())
