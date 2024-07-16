""" FLIP metric tool """
#################################################################################
# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################

# Visualizing and Communicating Errors in Rendered Images
# Ray Tracing Gems II, 2021,
# by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
# Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

# Visualizing Errors in Rendered High Dynamic Range Images
# Eurographics 2021,
# by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
# Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

# FLIP: A Difference Evaluator for Alternating Images
# High Performance Graphics 2020,
# by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
# Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
# Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

# Code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.

import numpy as np
import argparse
import time
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

from helper.flip_api import compute_ldrflip


def check_nans(reference, test, verbosity):
    """
    Checks reference and test images for NaNs and sets NaNs to 0. Depending on verbosity level, warns if NaNs occur

    :param reference: float tensor
    :param test: float tensor
    :param verbosity: (optional) integer describing level of verbosity.
                      0: no printed output,
                      1: print mean FLIP error,
                      "2: print pooled FLIP errors, PPD, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures"
                      3: print pooled FLIP errors, PPD, warnings, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures
    :return: two float tensors
    """
    if (np.isnan(reference)).any() or (np.isnan(test)).any():
        reference = np.nan_to_num(reference)
        test = np.nan_to_num(test)
        if verbosity == 3:
            print('=====================================================================')
            print('WARNING: either reference or test (or both) images contain NaNs.')
            print('Those values have been set to 0.')
            print('=====================================================================')
    return reference, test



def file_to_dsp(img):
    sos_img = np.sqrt((np.abs(np.fft.ifftn(img[..., ::2] + 1j * img[..., 1::2], axes=(0, 1))) ** 2).sum(axis=-1))
    return sos_img


def sense_to_dsp(img):
    # sos_img = np.sqrt((np.abs(img[..., 0] + 1j * img[..., 1])).sum(axis=0))
    sos_img = (np.abs(img[..., 0] + 1j * img[..., 1]))[0]
    return sos_img


if __name__ == '__main__':
    major_version = 1
    minor_version = 2

    hdr = False
    tone_mapper = None
    start_exposure = None
    stop_exposure = None
    num_exposures = None
    no_exposure_map = True

    import helper.misc as hmisc
    import helper.array_transf as harray
    import os
    ddata_target = '/home/bugger/Documents/data/7T/target'
    ddata_pred = '/home/bugger/Documents/data/7T/modeloutput'

    file_list = os.listdir(ddata_target)
    sel_pred = hmisc.load_array(os.path.join(ddata_pred, file_list[0]), sel_slice='mid', data_key='reconstruction')
    sel_target = hmisc.load_array(os.path.join(ddata_target, file_list[0]), sel_slice='mid', data_key='kspace')
    sel_target = file_to_dsp(sel_target)
    sel_target = harray.scale_minmax(sel_target)
    sel_pred = harray.scale_minmax(sel_pred)

    reference, test = check_nans(test=sel_pred, reference=sel_target, verbosity=0)
    import helper.plot_class as hplotc
    hplotc.ListPlot([sel_target, test])

    # Compute LDR-FLIP
    t0 = time.time()
    flip = compute_ldrflip(np.array([test, test, test]),
                           np.array([test, test, test])
                           ).squeeze(0)
    hplotc.ListPlot([flip], cbar=True)