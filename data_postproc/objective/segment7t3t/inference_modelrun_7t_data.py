import json
import helper.array_transf as harray
import helper.plot_class as hplotc
import objective.segment7T3T.executor_segment7T3T as executor
import objective.segment7T3T.recall_segment7T3T as recall
import nibabel
import os
import numpy as np
import torch

"""
Here I can execute MY model on the 7T data...

How did I do the nnUNet inference on 7T again...?
"""

ddata_7t = '/data/cmr7t3t/cmr7t/Image'
dmodel_run = '/data/seb/model_run'
file_list_7t = os.listdir(ddata_7t)

answer = None
for i_model_path in os.listdir(dmodel_run):
    print(f" ------------ Running model {i_model_path} -------------- ")
    ddest_result = f'/data/cmr7t3t/cmr7t/Results/{i_model_path}'
    if not os.path.isdir(ddest_result):
        os.makedirs(ddest_result)
    dconfig = os.path.join(dmodel_run, i_model_path)
    if answer != "All":
        print("Run inference?: All/Y/n")
        answer = input()
        print("Answer is: ", answer)

    if answer in ['Y', 'y', 'yes', 'Yes', 'All']:
        recall_obj = recall.RecallSegment3T7T(dconfig)
        recall_obj.mult_dict['config_00']['dir']['doutput'] = os.path.join(dconfig, "config_00")
        modelrun_obj = recall_obj.get_model_object(recall_obj.mult_dict['config_00'], load_model_only=True, inference=False)
        recall_obj.run_inference(file_list=file_list_7t, orig_dir=ddata_7t, dest_dir=ddest_result, modelrun_obj=modelrun_obj)
    else:
        print("Moving to the next model")
