"""
Lets see if we can do some tropical geometry
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.data
import scipy.signal
import helper.plot_class as hplotc


# Quick test for 2D numpy convolution
def convolve2D(image, kernel, padding=0, strides=1, dtype_output=np.float):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput), dtype=dtype_output)

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        result = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape])
                        output[x, y] = result.sum()
                except:
                    break

    return output


class Tropical(object):
    def __init__(self, x):
        self.x = x

    def __mul__(self, other):
        return Tropical(self.x + other.x)

    def __rmul__(self, other):
        return Tropical(self.x + other)

    def __add__(self, other):
        return Tropical(max(self.x, other.x))

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return str(self.x)


x1 = Tropical(3)
x2 = Tropical(5)

x_range = np.arange(-10, 10, 0.1)
y = x_range ** 2
plt.plot(x_range, y)

trop_x_range = np.array([Tropical(x) for x in x_range])
y_tropical = trop_x_range * trop_x_range
y_tropical = [x.x for x in y_tropical]
plt.plot(x_range, y_tropical)

A = np.array([x1, x1, x1]).reshape(3,1)
B = np.array([x2, x2, x2]).reshape(1,3)
Z = np.matmul(A, B)

n_kernel = 8
tropical_kernel = np.array([Tropical(np.random.randint(-15, -5)) for _ in range(n_kernel**2)]).reshape((n_kernel, n_kernel))
normal_kernel = np.array([(1/(n_kernel ** 2)) for _ in range(n_kernel**2)]).reshape((n_kernel, n_kernel))

tropical_kernel_line = np.array([Tropical(np.random.randint(-15, 15)) for _ in range(n_kernel)])
normal_kernel_line = np.array([(1/(n_kernel)) for _ in range(n_kernel)])

A = skimage.data.astronaut()
A_women = A[:, :, 0]
A_women_tropical = np.reshape([Tropical(x) for x in A_women.ravel()], A_women.shape)

A_women_line = A_women[A_women.shape[0]//2]
A_normal_line = np.convolve(A_women_line, normal_kernel_line)
A_tropical_line = np.convolve([Tropical(x) for x in A_women_line], tropical_kernel_line)

plt.figure()
plt.plot(A_women_line, 'k', label='original')
plt.plot(A_normal_line, 'r', label='normal')
plt.plot([x.x for x in A_tropical_line], 'b', label='tropical')
plt.legend()

A_women_smooth_normal = convolve2D(A_women, normal_kernel)
# This takes some time.. totally worth it...
A_women_smooth_tropical = convolve2D(A_women_tropical, tropical_kernel, dtype_output=Tropical)
A_women_smooth_tropical = np.reshape([x.x for x in A_women_smooth_tropical.ravel()], A_women_smooth_tropical.shape)

hplotc.ListPlot([A_women, A_women_smooth_tropical, A_women_smooth_normal], ax_off=True, subtitle=[['Original'], ['Tropical Smooth'], ['Normal smooth']])
plt.imshow(A_women_smooth_normal)
plt.imshow(A_women_smooth_tropical)

# Now check if we can do this faster with line by line.. Not really doable since you really want 2D cnvs..?