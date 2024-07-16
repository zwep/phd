import os
import data_generator.Segment7T3T as dg_segment_7t3t
import helper.array_transf
import helper.array_transf as harray
import helper.misc as hmisc
import numpy as np
import matplotlib.pyplot as plt
import torchio.transforms

"""
Check the distributions of both the torchIO biasfield as well as my own generated biasfields...
"""


ddata_biasf = '/data/cmr7t3t/biasfield_sa_mm1_A'
ddata_gan = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task903_GAN_MM1_A_balanced/imagesTr'
ddata_mm1a = '/data/seb/nnunet/nnUNet_raw/nnUNet_raw_data/Task501_MM1_A/imagesTr'
file_list_mm1 = os.listdir(ddata_mm1a)

dataset_type = 'train'  # Change this to train / test
dg_obj = dg_segment_7t3t.DataGeneratorCardiacSegment(ddata=ddata_biasf,
                                                     dataset_type=dataset_type, target_type='biasfield',
                                                     transform_resize=True,
                                                     transform_type="abs")

n_fields = 1

sim_biasfield_list = []
for ii in range(n_fields):
    print(ii)
    container = dg_obj.__getitem__(ii)
    biasfield_array = container['target'].numpy()
    biasfield_values = [x for x in biasfield_array.ravel() if x > 1e-3]
    sim_biasfield_list.extend(biasfield_values)


gan_biasfield_list = []
for ii in range(n_fields):
    print(ii)
    sel_file = file_list_mm1[ii]
    inhomog_file = os.path.join(ddata_gan, sel_file)
    homog_file = os.path.join(ddata_mm1a, sel_file)
    inhomog_image = hmisc.load_array(inhomog_file)[:, :, 0]
    inhomog_image = harray.scale_minmax(inhomog_image)
    homog_image = hmisc.load_array(homog_file)[:, :, 0]
    homog_image = harray.scale_minmax(homog_image)
    biasf_image = np.squeeze(inhomog_image / homog_image)
    biasf_image = helper.array_transf.correct_inf_nan(biasf_image)
    biasf_image = harray.scale_minmax(biasf_image)
    biasfield_values = [x for x in biasf_image.ravel() if x > 1e-3]
    gan_biasfield_list.extend(biasfield_values)

# Torch Biasfield object
poly_2_biasfield_list = []
dummy_array = np.ones((256, 256))
for _ in range(n_fields):
    print(_)
    gen_biasf_obj = torchio.transforms.RandomBiasField(coefficients=0.8, order=2)
    gen_biasf = gen_biasf_obj(dummy_array[None, :, :, None])[0, :, :, 0]
    gen_biasf = harray.scale_minmax(gen_biasf)
    poly_2_biasfield_list.extend(gen_biasf.ravel())

poly_5_biasfield_list = []
for _ in range(n_fields):
    gen_biasf_obj = torchio.transforms.RandomBiasField(coefficients=0.8, order=5)
    gen_biasf = gen_biasf_obj(dummy_array[None, :, :, None])[0, :, :, 0]
    gen_biasf = harray.scale_minmax(gen_biasf)
    poly_5_biasfield_list.extend(gen_biasf.ravel())

fig, ax = plt.subplots(2,2)
ax = ax.ravel()
_ = ax[0].hist(sim_biasfield_list, label='simulated', bins=256)
ax[0].set_title('distr of simulated bias field values')
_ = ax[1].hist(poly_2_biasfield_list, label='polynomial', bins=256)
ax[1].set_title('distr of polynomial 2 bias field values')
_ = ax[2].hist(poly_5_biasfield_list, label='polynomial', bins=256)
ax[2].set_title('distr of polynomial 5 bias field values')
_ = ax[3].hist(gan_biasfield_list, label='GAN', bins=256)
ax[3].set_title('distr of GAN based bias field values')
fig.tight_layout()
fig.savefig('/data/seb/test_distr.png')


## Misc

dbaby = '/home/bugger/Documents/data/7T/baby_names'
with open(dbaby, 'r') as f:
    A = [x.strip().split(",") for x in f.readlines()]

import matplotlib.pyplot as plt
import numpy as np
mean_weight = np.mean([int(x[-1]) for x in A if x[-1]])
mean_length = np.mean([float(x[-2]) for x in A if x[-2].strip()])
hour_list, minute_list = zip(*[x[-3].split(".") for x in A if x[-3].strip()])
mean_hour = np.mean([int(x) for x in hour_list])
mean_minute = np.mean([int(x) for x in minute_list])

all_letter_list = [list(x[0].lower()) for x in A]
max_letters = max([len(x) for x in all_letter_list])
for i_letters in all_letter_list:
    pad_null = max_letters - len(i_letters)
    i_letters.extend([""] * pad_null)
import helper.misc as hmisc
shifted_names = hmisc.change_list_order(all_letter_list)
''.join([np.random.choice(x) for x in shifted_names])

mean_weight
mean_length