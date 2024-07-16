import helper.array_transf
from objective.postproc_base import PostProcBase
import cv2
import nibabel
import helper.array_transf as harray
import skimage.transform as sktransform
from skimage.util import img_as_ubyte
import helper.misc as hmisc
import helper.plot_class as hplotc
import torch
import numpy as np
import h5py
import os
import tensorflow as tf
from data_generator.Generic import transform_array
from loguru import logger
from objective_configuration.inhomog_removal import LOG_DIR, INHOMONET_PATH, INHOMONET_WEIGHTS

try:
    file_base_name = hmisc.get_base_name(__file__)
    logger.add(os.path.join(LOG_DIR, f"{file_base_name}.log"))
except NameError:
    print('No file name known. Not reporting to logger.')


class PostProcReconstruction(PostProcBase):
    def __init__(self, executor_module, config_dict=None, config_path=None, config_name='config_run.json', **kwargs):
        super().__init__(executor_module=executor_module, config_dict=config_dict, config_path=config_path, config_name=config_name)
        pass
