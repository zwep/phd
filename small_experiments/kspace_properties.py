import numpy as np
import helper.array_transf as harray
import helper.plot_class as hplotc
import skimage.data
import skimage.transform as sktransf

A = skimage.data.astronaut()[:, :, 0]
A_kspace = harray.transform_image_to_kspace_fftn(A)
A_kspace = np.pad(A_kspace, [(0, 1), (0, 1)])
A = harray.transform_kspace_to_image_fftn(A_kspace)
A_kspace = harray.transform_image_to_kspace_fftn(A)
N = A.shape[0]
hplotc.ListPlot([A, A_kspace], augm='np.abs')

# Test resizing with padding in kspace.. quite funny
A_kspace_padded = np.pad(A_kspace, [(100, 100), (200, 200)])
A_padded = harray.transform_kspace_to_image_fftn(A_kspace_padded)
A_padded = (harray.scale_minmax(np.abs(A_padded)) * 255).astype(int)
A_resize = sktransf.resize(A, A_padded.shape, preserve_range=True, anti_aliasing=False)
hplotc.ListPlot([A_padded, A_resize, np.abs(A_padded) - A_resize], augm='np.abs', cbar=True)

# Voorbeeld met 10...
# 1 2 3 4 | 5 | 6 | 7 8 9 10
# 0 1 2 3 | 4 | 5 | 6 7 8 9
# Nu zijn lijn 5 en 6 de 'middelste', of index 4 en 5 (== N//2-1 en N//2)
#
# Tring to switch halves first...
B_kspace = np.zeros((N, N), dtype=complex)
# Since N is even we need to do this...
# --> We made N odd, so we have it easier?
import matplotlib.pyplot as plt
plt.plot(np.abs(A_kspace[N//2]))
plt.plot(np.abs(A_kspace[N//2-1]))

# First quartile
quartile_1 = A_kspace[:N//2, N//2+1:]
# Third quartile..
quartile_3 = A_kspace[N//2+1:, :N//2][::-1, ::-1]
# he..he.. dit is wat ik wil..
hplotc.ListPlot([quartile_1, quartile_3, quartile_1 - quartile_3.conjugate()], augm='np.abs', vmin=(0, 1000))

B_kspace[:, :N//2-1] = A_kspace[:, :N//2-1]
# Copy the middle two lines since N is even
B_kspace[:, N//2-1] = A_kspace[:, N//2-1]
B_kspace[:, N//2] = A_kspace[:, N//2]
# Now copy the top half to the bottom
B_kspace[:, N//2+1:] = B_kspace[:, :N//2-1].conjugate()[::-1, ::-1]
hplotc.ListPlot(B_kspace, augm='np.abs', vmin=(0, 2000))
B_1 = harray.transform_kspace_to_image_fftn(B_kspace)
hplotc.ListPlot(np.abs(B_1))
# Switching halves on different axis
B_kspace = np.zeros((N, N), dtype=complex)
# Since N is even we need to do this...
B_kspace[:N//2-1, :] = A_kspace[:N//2-1, :]
B_kspace[N//2, :] = A_kspace[N//2, :]
B_kspace[N//2+1:, :] = B_kspace[:N//2-1, :].conjugate()[::-1, ::-1]
B_2 = harray.transform_kspace_to_image_fftn(B_kspace)

hplotc.ListPlot([A, B_1, B_2, np.sqrt(B_1**2 + B_2**2)], augm='np.abs')

# Now try quantiles...

# Tring to switch halves first...
B_kspace = np.zeros((N, N), dtype=complex)
# Since N is even we need to do this...
B_kspace[:N//2, :N//2] = A_kspace[:N//2, :N//2]
B_kspace[:, N//2] = A_kspace[:, N//2]
# B_kspace[:, N//2+1:] = np.fft.ifftshift(B_kspace[:, :N//2-1].conjugate(), axes=-1)
B_kspace[:, N//2+1:] = B_kspace[:, :N//2-1].conjugate()[::-1, ::-1]
B_1 = harray.transform_kspace_to_image_fftn(B_kspace)