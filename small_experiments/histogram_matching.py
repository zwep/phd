"""
Small experiment for histogram equilization

Based on what I read at
https://en.wikipedia.org/wiki/Histogram_matching

Because other code that I used was of no use...
"""


import matplotlib.pyplot as plt
import numpy as np
import skimage.data


A = skimage.data.astronaut()[:, :, 0]
B = skimage.data.camera()

A_hist_obj = plt.hist(A.ravel(), bins=256, density=True)
A_hist_pdf = A_hist_obj[0]
A_hist_cdf = [sum(A_hist_pdf[:i]) for i in range(len(A_hist_pdf))]

B_hist_obj = plt.hist(B.ravel(), bins=256, density=True)
B_hist_pdf = B_hist_obj[0]
B_hist_cdf = np.array([sum(B_hist_pdf[:i]) for i in range(len(B_hist_pdf))])

fig, ax = plt.subplots(2)
ax[0].imshow(A)
ax[1].imshow(B)

plt.figure()
plt.plot(A_hist_pdf)
plt.plot(B_hist_pdf)

# Match dem histgroams
for pixel_value, value_hist in enumerate(A_hist_cdf):
    # So now we have the pixel value we want....
    index_B = np.argmin(np.abs(B_hist_cdf - value_hist))
    A[A == pixel_value] = index_B

fig, ax = plt.subplots(2)
ax[0].imshow(A)
ax[1].imshow(B)

plt.figure()
plt.hist(A.ravel(), bins=256)
plt.hist(B.ravel(), bins=256)

from skimage.exposure import match_histograms
reference = skimage.data.coffee()
image = skimage.data.chelsea()


image = skimage.data.astronaut()[:, :, 0]
reference = skimage.data.camera()

matched = match_histograms(image, reference, multichannel=False)

fig, ax = plt.subplots(3)
ax[0].imshow(image)
ax[0].set_title('Source')
ax[1].imshow(reference)
ax[1].set_title('Reference')
ax[2].imshow(matched)
ax[2].set_title('Matched')

