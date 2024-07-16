"""
lets check out this package..
"""

import mridc
import omegaconf
from mridc.collections.reconstruction.models import ccnn
conf = omegaconf.OmegaConf.create({"coil_combination_method": "RSS",
                                   "fft_centered": True,
                                   "fft_normalization": True,
                                   "spatial_dims": (0, 1),
                                   "coil_dim": 2,
                                   "hidden_channels": 2,
                                   "n_convs": 1,
                                   "batchnorm": True,
                                   "no_dc": True,
                                   "num_cascades": 2,
                                   "train_loss_fn": "ssim",
                                   "eval_loss_fn": "ssim"})



from pytorch_lightning import Trainer
trainer = Trainer()
model_obj = ccnn.CascadeNet(conf, trainer=trainer)
model_obj.trainer.fit(model_obj)

# Can I later load the weights from DCNN...?
# And use them here?
import torch
import numpy as np
A = np.random.rand(1, 100, 100)
A_tens = torch.from_numpy(A).float()
sens = torch.from_numpy(A).float()
mask = torch.from_numpy(A).float()
init = torch.from_numpy(A).float()
target = torch.from_numpy(A).float()
model_obj(A_tens, A_tens, A_tens, A_tens, A_tens)

from mridc.collections.reconstruction.models import vsnet

