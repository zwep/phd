"""
Classical model

PABIC - PArametric BIas field Correction

Juntu J., Sijbers J., Van Dyck D., Gielen J. (2005) Bias Field Correction for MRI Images.
In: Kurzyński M., Puchała E., Woźniak M., żołnierek A. (eds)
Computer Recognition Systems. Advances in Soft Computing, vol 30. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/3-540-32390-2_64

The steps of the algorithm are:

1. Extract a background image from the corrupted MRI image, for example, by smoothing the image with a Gaussian filter
of a large bandwidth (about 2/3 the size of the MRI image) to filter out all the image details that correspond
to high- frequency components.
2. Select few data points from the background image and save their coordinates and graylevel values into
a matrix D = (xi, yi, gi), i = 1, 2, ...n.
It is recommended not to select points from the regions where there is no MRI signal since this regions has no bias field signal.

3. Select a parametric equation for the fitted surface . It is better to fit simple surfaces such as
low order polynomial surfaces since they are very smooth and their parameters are very easy to estimate.

4. Estimate the parameters of the surface that best fits the data in matrix D by the method of nonlinear least-squares.
5. Use the fitted equation to generate an image of the bias field signal.
6. Divide the corrupted MRI image by the estimated bias field image in step 5.

"""

import os
import numpy as np
import scipy.optimize
import helper.array_transf as harray
import helper.plot_class as hplotc
import helper.plot_fun as hplotf
import scipy.signal


def get_lst_sqr(x, mask=None, *kwargs):
    # Fit a polynomial to your own data
    x = harray.scale_minmax(x)
    # Get a mask..
    if mask is None:
        mask_array = harray.get_treshold_label_mask(x)
    else:
        mask_array = mask

    # Smooth the image...
    n_kernel = x.shape[0]//10
    smooth_kernel = np.ones((n_kernel, n_kernel)) / n_kernel**2
    image_smooth = scipy.signal.convolve2d(x, smooth_kernel, mode='same')
    image_smooth = harray.scale_minmax(image_smooth)

    # What we want to approximate with a polynomial..
    D = mask_array * image_smooth

    ny, nx = D.shape
    x_range = np.linspace(-1, 1, nx)
    y_range = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x_range, y_range)
    X_D = (X * mask_array).ravel()
    Y_D = (Y * mask_array).ravel()

    # Optimization matrix
    Z = np.array([np.ones(ny*nx), X_D, X_D ** 2, X_D ** 3, Y_D, Y_D ** 2, Y_D ** 3]).T
    x_opt = np.matmul(np.linalg.pinv(Z), D.ravel())
    D_approx = np.matmul(Z, x_opt).reshape(ny, nx)
    return x / D_approx


if __name__ == "__main__":
   # Here we can test out this procedure....
    # Using measured data (read cpx)

    measured_path = '/home/bugger/Documents/data/7T/test_for_inhomog/prostate_7T'
    file_list = [os.path.join(measured_path, x) for x in os.listdir(measured_path)]
    sel_file_list = [x for x in file_list if x.endswith('npy')]

    for i_file in sel_file_list[:13]:
        A = np.load(i_file)
        A_abs = np.abs(A.sum(axis=0))
        A_abs = harray.scale_minmax(A_abs)

        D_approx = get_lst_sqr(A_abs)
        hplotf.plot_3d_list(D_approx, vmin=(0,1))

        hplotc.ListPlot([[A_abs, A_abs / D_approx]])