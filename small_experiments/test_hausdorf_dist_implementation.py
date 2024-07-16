import torch

import skimage.transform
import importlib
import time
import h5py
import helper.array_transf as harray
import helper.plot_class as hplotc
import helper_torch.loss as hloss
import numpy as np

"""
Implementing an efficinet hausdorf distance
"""

dprostate = '/home/bugger/Documents/data/1.5T/prostate_mri_mrl/4_MR/MRL/20201231_0002.h5'

with h5py.File(dprostate, 'r') as f:
    A = np.array(f['data'][30])
    B = np.array(f['data'][-1])

A_mask = harray.get_treshold_label_mask(A)
B_mask = harray.get_treshold_label_mask(B)

N = 128
A_mask = skimage.transform.resize(A_mask, (N, N), preserve_range=True, anti_aliasing=False)
B_mask = skimage.transform.resize(B_mask, (N, N), preserve_range=True, anti_aliasing=False)
A_mask_tens = torch.from_numpy(A_mask).float()
B_mask_tens = torch.from_numpy(B_mask).float()

A_mask_indx = np.argwhere(A_mask == 1)
B_mask_indx = np.argwhere(B_mask == 1)


loss_obj = hloss.BalancedAveragedHausdorffLoss()
t0 = time.time()
loss_value = loss_obj(A_mask_tens[None, None], B_mask_tens[None, None])
print('time ', time.time() - t0)

print('')

loss_obj = hloss.EfficientBalancedAveragedHausdorffLoss()
t0 = time.time()
loss_value = loss_obj(A_mask_tens[None, None], B_mask_tens[None, None])
print('time ', time.time() - t0)

# TODO MAybe test here something scaled with an exp(-x)
x = A_mask_tens[None, None]
y = B_mask_tens[None, None]
x_shape = x.shape
xy_shape = x_shape[-2:]
n_batch_chan = torch.prod(torch.tensor(x_shape[:2]))

# Reshape pred and target into (-1, x, y)
pred = x.reshape((-1, *xy_shape))
target = y.reshape((-1, *xy_shape))
batch_pred_loc, x_pred_loc, y_pred_loc = torch.where(pred)
pred_loc = torch.stack([x_pred_loc, y_pred_loc], dim=-1)
batch_tgt_loc, x_tgt_loc, y_tgt_loc = torch.where(target)
tgt_loc = torch.stack([x_tgt_loc, y_tgt_loc], dim=-1)

loss_value = 0
for i in range(0, n_batch_chan):
    n_points_ground_truth = torch.sum(batch_tgt_loc == i)
    differences = pred_loc[batch_pred_loc == i].unsqueeze(1) - tgt_loc[batch_tgt_loc == i].unsqueeze(0)
    distances = torch.sum(torch.abs(differences.float()) ** 2, -1) ** (1 / 2)
    hplotc.ListPlot(distances[None])


min_distances, min_indices = torch.min(distances, 1)
import matplotlib.pyplot as plt
plt.plot(min_distances)
#       print('n_points_ground_truth', n_points_ground_truth)
#      print('Min distances ', min_distances.shape, min_distances)
term_1 = torch.sum(min_distances) / n_points_ground_truth