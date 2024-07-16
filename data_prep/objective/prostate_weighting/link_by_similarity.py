import helper.array_transf as harray
from skimage.metrics import structural_similarity
import itertools
import multiprocessing as mp
import skimage.transform as sktransf
import numpy as np
import os
import data_generator.ProstateWeighting as data_gen
import helper.plot_class as hplotc
import h5py

"""
A linear relation does not really exist bewteen the input/target (1.5T/3T) images.
So we need to do something else
Probably something with a similarity

This means the following..
1.5T image has shape (n_slice, nx, ny)
3T image has shape (m_slice, mx, my)

Get a n_slice x m_slice matric to compare each image with eachother
SSIM is symmetric so that makes stuff easier I guess

However...this also didnt quite did the job
Finally I created a class CompareImages in adifferent file that wenth through them based on DICE score on the masks
and on the SSIM ...
"""


# Here we are going to check the current level of similarity between the input and target images.
dg_obj = data_gen.DataGeneratorProstateWeighting(ddata='/local_scratch/sharreve/mri_data/tconvert_h5', debug=True, dataset_type='validation')
sel_dataset = 0
file_list = dg_obj.container_file_info[sel_dataset]['file_list']
input_dir = dg_obj.container_file_info[sel_dataset]['input_dir']
target_dir = dg_obj.container_file_info[sel_dataset]['target_dir']
mask_dir = dg_obj.container_file_info[sel_dataset]['mask_dir']
index = 0
i_file = file_list[index]
file_name, ext = os.path.splitext(i_file)
input_file = os.path.join(input_dir, i_file)
target_file = os.path.join(target_dir, i_file)


with h5py.File(input_file, 'r') as h5_obj:
    input_array = np.array(h5_obj['data'])

with h5py.File(target_file, 'r') as h5_obj:
    target_array = np.array(h5_obj['data'])

counter = 0
max_slice_input = input_array.shape[0]
max_slice_target = target_array.shape[0]

min_shape = min(input_array.shape[-2:], target_array.shape[-2:])
target_array = sktransf.resize(target_array, (max_slice_target, ) + min_shape, preserve_range=True)
input_array = sktransf.resize(input_array, (max_slice_input, ) + min_shape, preserve_range=True)

input_array = harray.scale_minmax(input_array)
target_array = harray.scale_minmax(target_array)

N = mp.cpu_count()
parallel_input = np.arange(0, 100)
print('Amount of CPUs ', N)
print('Amount of iterations ', len(parallel_input))

# temp = [itertools.product(range(x, max_slice_input), [x]) for x in range(max_slice_target)]
# ij_coords = list(itertools.chain(*temp))

ij_coords = list(itertools.product(range(max_slice_input), range(max_slice_target)))

def get_ssim(inp):
    i, j = inp
    temp_input = input_array[i]
    temp_target = target_array[j]
    # dist = structural_similarity(temp_input, temp_target)
    dist = ((temp_input - temp_target) ** 2).sum()
    return dist

N = mp.cpu_count()
N = N // 2
with mp.Pool(processes=N) as p:
    results = p.map(get_ssim, ij_coords)

# Now map them back...
distance = np.zeros((max_slice_input, max_slice_target))
for ij, dist_value in zip(ij_coords, results):
    i, j = ij
    distance[i, j] = dist_value

# Finally, this comparisson gave us a large covariance matrix of a sort
plot_obj = hplotc.ListPlot(distance, cbar=True)
plot_obj.figure.savefig('/local_scratch/sharreve/distance.jpg', bbox_inches='tight')

nice_duo = []
for ii, i in enumerate(distance):
    j = np.argmax(i)
    nice_duo.append((ii, j))

counter = 0
for i_batch in np.split(np.array(nice_duo), 41):
    counter += 1
    plot_array = [[input_array[x], target_array[y]] for x, y in i_batch]
    plot_obj = hplotc.ListPlot(plot_array)
    plot_obj.figure.savefig(f'/local_scratch/sharreve/example_{counter}.jpg', bbox_inches='tight')


"""
Since the previous method did not gave any satisfactorial restuls.. 
 we are just going to check the first, middle and last slice... and get a dice score for overlap and a similarity score
"""

from helper.metric import dice_score


def get_score(x_img, y_img, x_mask=None, y_mask=None):
    if x_mask is None:
        x_mask = harray.get_treshold_label_mask(x_img)
    if y_mask is None:
        y_mask = harray.get_treshold_label_mask(y_img)
    x_img = harray.scale_minmax(x_img)
    y_img = harray.scale_minmax(y_img)
    dice_value = dice_score(x_mask.astype(bool), y_mask.astype(bool))
    ssim_value = structural_similarity(x_img, y_img)
    return dice_value, ssim_value

dg_obj = data_gen.DataGeneratorProstateWeighting(ddata='/local_scratch/sharreve/mri_data/tconvert_h5', debug=True, dataset_type='train')
sel_dataset = 0
file_list = dg_obj.container_file_info[sel_dataset]['file_list']
input_dir = dg_obj.container_file_info[sel_dataset]['input_dir']
target_dir = dg_obj.container_file_info[sel_dataset]['target_dir']

counter = 0
for i_file in file_list:
    counter += 1
    file_name, ext = os.path.splitext(i_file)
    print(file_name)
    input_file = os.path.join(input_dir, i_file)
    target_file = os.path.join(target_dir, i_file)
    with h5py.File(input_file, 'r') as h5_obj:
        start_input_array = np.array(h5_obj['data'][0])
    with h5py.File(target_file, 'r') as h5_obj:
        start_target_array = np.array(h5_obj['data'][0])
    with h5py.File(input_file, 'r') as h5_obj:
        n_max = h5_obj['data'].shape[0]
        mid_input_array = np.array(h5_obj['data'][n_max//2])
    with h5py.File(target_file, 'r') as h5_obj:
        n_max = h5_obj['data'].shape[0]
        mid_target_array = np.array(h5_obj['data'][n_max//2])
    with h5py.File(input_file, 'r') as h5_obj:
        end_input_array = np.array(h5_obj['data'][-1])
    with h5py.File(target_file, 'r') as h5_obj:
        end_target_array = np.array(h5_obj['data'][-1])
    plot_obj = hplotc.ListPlot([[start_input_array, start_target_array],
                                [mid_input_array, mid_target_array],
                                [end_input_array, end_target_array]], title=f'{file_name}')
    plot_obj.figure.savefig(f'/local_scratch/sharreve/example_{counter}.jpg', bbox_inches='tight')


# These were OKAY in the train set --> But I dont know anymore what I did with them....
MR_patients = [20, 39, 13, 11, 42, 34, 4, 6, 1, 47, 9, 12, 37, 2, 32, 38, 27, 44]
MR_patients = [str(x) + "_MR" for x in MR_patients]
counter = 0
for i_file in file_list:
    counter += 1
    file_name, ext = os.path.splitext(i_file)
    if file_name not in MR_patients:
        print(file_name)
        input_file = os.path.join(input_dir, i_file)
        target_file = os.path.join(target_dir, i_file)
        with h5py.File(input_file, 'r') as h5_obj:
            print(h5_obj['data'].shape[0])
        with h5py.File(target_file, 'r') as h5_obj:
            print(h5_obj['data'].shape[0])

            # 3 17 14?