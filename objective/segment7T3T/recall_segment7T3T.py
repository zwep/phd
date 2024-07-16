
import time
from skimage.util import img_as_ubyte, img_as_int, img_as_uint
import scipy.spatial.distance
import scipy.stats
import argparse
import os
import torch
import helper.misc as hmisc
import helper.array_transf as harray
import objective.segment7T3T.executor_segment7T3T as executor
from objective.recall_base import RecallBase
import nibabel
import numpy as np

class RecallSegment3T7T(RecallBase):

    def get_model_object(self, config_file=None, load_model_only=False, inference=True):
        decision_obj = executor.DecisionMakerSegment7T3T(config_file=config_file, debug=False,
                                                         load_model_only=load_model_only, inference=inference, device=None)  # ==>>
        modelrun_obj = decision_obj.decision_maker()
        modelrun_obj.device = torch.device('cpu')
        network_choice = hmisc.get_nested(modelrun_obj.config_param, ['model', 'config_regular', 'network_choice'])
        config_model = hmisc.get_nested(modelrun_obj.config_param, ['model', f'config_{network_choice}'])
        modelrun_obj.model_obj = modelrun_obj.get_model(config_model=config_model, model_choice=network_choice)
        modelrun_obj.load_weights()
        return modelrun_obj

    def run_inference(self, file_list, orig_dir, dest_dir, modelrun_obj):
        # This is very specific for this task.. since we are dealing with nifti-files and storing them
        counter = -1
        for sel_file in file_list:
            sel_file_path = os.path.join(orig_dir, sel_file)
            dest_file_path = os.path.join(dest_dir, sel_file)
            counter += 1
            nibabel_object = nibabel.load(sel_file_path)
            sel_array = np.array(nibabel_object.get_fdata())
            sel_array = np.moveaxis(sel_array, -1, 0)
            result = []
            for i_card in sel_array:
                i_card = harray.scale_minmax(i_card)
                torch_tensor = torch.from_numpy(i_card[None, None]).float()
                with torch.no_grad():
                    x = modelrun_obj.model_obj(torch_tensor)
                x_padded = np.concatenate([np.zeros(x.shape[-2:])[None], x.numpy()[0]])
                # Here are some hyper-parameters that can influence the outcome.
                x_rounded = np.isclose(x_padded, 1, atol=0.8).astype(int)
                x_maxed = np.argmax(x_rounded, axis=0)
                result.append(x_maxed)

            result = np.moveaxis(np.array(result), 0, -1)
            nibabel_result = nibabel.Nifti1Image(result, nibabel_object.affine)
            nibabel.save(nibabel_result, dest_file_path)
