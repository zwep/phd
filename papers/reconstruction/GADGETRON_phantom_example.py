import h5py
from matplotlib import pyplot as plt
import numpy as np

f = h5py.File('/home/bugger/PycharmProjects/gadgetron_conda/out.h5')
image_series = list(f.keys())
data = np.array(f[image_series[0]]['image_0']['data']).squeeze()
plt.imshow(data[0,:,:])
f.close()
