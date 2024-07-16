import numpy as np
import tensorflow
from skimage.metrics import structural_similarity
import skimage.data

A = skimage.data.astronaut()[:, :, 0]
B = skimage.data.camera()

structural_similarity(A, B)
structural_similarity(A, B, data_range=255)
structural_similarity(A * 1., B * 1.)
structural_similarity(A * 1., B * 2., data_range=255)


C = (B * 2.).astype(np.uint8)   # Derp this converts stuff to negative values
from skimage.util import img_as_ubyte, img_as_uint, img_as_int
img_as_ubyte(B * 2.)
tensorflow.image.ssim(A[None] * 1., B[None] * 2., max_val=255)
(C * 1.).min()