
"""
There are some parameters that need to be set properly. Especially for binary masks
"""

import numpy as np
import skimage.transform as sktransf
import helper.plot_class as hplotc
import skimage.data

# Example binary mask
A = np.random.randint(0, 2, size=(50, 50))
B = sktransf.resize(A, (100, 100), preserve_range=True)
B1 = sktransf.resize(A, (25, 25), preserve_range=True, anti_aliasing=False)
C = sktransf.resize(A, (100, 100), preserve_range=False)

hplotc.ListPlot([A, 1 - (B < 0.5), 1 - (B1 < 0.5), C])

A = skimage.data.astronaut()[:, :, 0]
B = sktransf.resize(A, (100, 100), preserve_range=True)
B1 = sktransf.resize(A, (25, 25), preserve_range=True, anti_aliasing=False)
B2 = sktransf.resize(A, (25, 25), preserve_range=True)
C = sktransf.resize(A, (100, 100), preserve_range=False)

hplotc.ListPlot([A, B, B1, B2, C], cbar=True)
