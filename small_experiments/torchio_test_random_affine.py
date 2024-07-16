import helper.plot_class as hplotc
import torchio.transforms
import torch
import numpy as np
import skimage.data


A = skimage.data.astronaut()[:, :, 0:1]
A_tens = torch.from_numpy(np.array([A]))
torchio_obj = torchio.transforms.RandomAffine(scales=[0.25, 4], translation=[-50, 50, -50, 50, 0, 0], degrees=[-10, 10, -10, 10, 0, 0],
                                isotropic=True, default_pad_value=0, center='origin')

transformed_A = []
for _ in range(25):
    A_transf = torchio_obj(A_tens)
    transformed_A.append(A_transf[:, :, :, 0].numpy())

hplotc.SlidingPlot(np.array(transformed_A))
