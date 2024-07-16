import torchio.transforms
import numpy as np
import helper.plot_class as hplotc
import helper.misc as hmisc
import helper.array_transf as harray
import matplotlib.pyplot as plt
from objective_configuration.segment7T3T import DCMR7T, DFINAL, DLOG
import os
from loguru import logger

logger.add(os.path.join(DLOG, 'compare_torch_io_and_biasfield.log'))
ddata_b1p = os.path.join(DCMR7T, 'b1_distr/sa/b1_plus/V1.npy')
array_b1p = hmisc.load_array(ddata_b1p)
logger.debug(f'Shape of loaded array is {array_b1p.shape}')
b1_array = np.abs(hmisc.load_array(ddata_b1p)).sum(axis=0)
b1_array = harray.scale_minmax(b1_array)
logger.debug(f'Processed array shape {b1_array.shape}')

data_shape = b1_array.shape
generated_biasfield = []
for i_order in [2, 5]:
    logger.debug(f'Generating biasfield with order {i_order}')
    for i_coeff in np.arange(0.1, 1.1, 0.1):
        temp_array = []
        for _ in range(10):
            gen_biasf = torchio.transforms.RandomBiasField(coefficients=i_coeff, order=i_order)
            # Generate a bias field on /nes..
            result = gen_biasf(np.ones(data_shape)[None, :, :, None])[0, :, :, 0]
            temp_array.append(result)
        generated_biasfield.append((i_order, i_coeff, temp_array))

for ii in range(len(generated_biasfield)):
    container = generated_biasfield[ii]
    logger.debug(f' Content of container {container[0]}, {container[1]}')
    i_example = container[2]
    logger.debug(f'Plotting poly-biasfield with order {i_order}')
    dest_dir = os.path.join(DFINAL, f'biasfield_tryout/example_{ii}.png')
    fig_obj = hplotc.ListPlot([i_example[0]], title=f'order {i_order} i coeff {i_coeff}')
    fig_obj.figure.savefig(dest_dir)
    hplotc.close_all()


# Now create histogram plots...
n_images = len(generated_biasfield) + 1
nx, ny = hmisc.get_square(n_images)
fig, ax = plt.subplots(nrows=nx, ncols=ny, figsize=(15, 15))
ax = ax.ravel()
# Gather all the B1p files
all_b1p_array = []
for ii in range(1, 14):
    ddata_b1p = os.path.join(DCMR7T, f'b1_distr/sa/b1_plus/V{ii}.npy')
    array_b1p = hmisc.load_array(ddata_b1p)
    logger.debug(f'Shape of loaded array is {array_b1p.shape}')
    b1_array = np.abs(hmisc.load_array(ddata_b1p)).sum(axis=0)
    b1_array = harray.scale_minmax(b1_array)
    all_b1p_array.append(b1_array)

for x in all_b1p_array:
    logger.debug(f'Shape of b1p array {x.shape}')

all_b1p_array = np.concatenate([x.ravel() for x in all_b1p_array])
non_zero_b1p = all_b1p_array[all_b1p_array != 0]
logger.debug(f'Shape of all b1p array {all_b1p_array.shape} {all_b1p_array.ravel().shape}')
ax[0].hist(((non_zero_b1p - np.mean(non_zero_b1p)) / np.std(non_zero_b1p) + 1).ravel(), bins=256, range=(0, 2), label='Biasfield histogram')
for ii in range(0, n_images-1):
    (i_order, i_coeff, i_example) = generated_biasfield[ii]
    ax[ii+1].hist(np.array(i_example).ravel(), bins=256, range=(0, 2), label=f'Order {np.round(i_order)} Coeff {np.round(i_coeff, 2)}')
    ax[ii+1].legend()

fig.savefig(os.path.join(DFINAL, 'biasfield_tryout/histogram_of_biasfield.png'))