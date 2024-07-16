import numpy as np
import helper.array_transf as harray
import os
import helper.misc as hmisc
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
from loguru import logger
from objective_configuration.segment7T3T import DLOG, DFINAL

"""
Im too nerdy to copy paste stuff
 
We can automate this.

For now I only visualize the effect of the augmentations... 
"""

logger.add(os.path.join(DLOG, "dataset_example.log"))

raw_data_path = '/home/bme001/20184098/data/nnunet/nnUNet_raw/nnUNet_raw_data'
dest_path = os.path.join(DFINAL, 'final_figures')

if not os.path.isdir(dest_path):
    os.makedirs(dest_path)

# This is so stupid.. put at least error proof..
# no augm dict
normal_dict = {'mm1a': 'Task501_MM1_A/imagesTr/A0S9V9_loc_00_0000.nii.gz',
               'mm1b': 'Task502_MM1_B/imagesTr/A1D0Q7_loc_00_0000.nii.gz',
               'acdc': 'Task511_ACDC/imagesTr/patient001_frame01_0000.nii.gz'}  # Here we need to select the 5th location

# Biafield sim
biasfield_sim_dict = {'mm1a': 'Task901_Biasfield_MM1_A_balanced/imagesTr/V12_to_A0S9V9_loc_00_0000.nii.gz',
                      'mm1b': 'Task902_Biasfield_MM1_B_balanced/imagesTr/V6_to_A1D0Q7_loc_00loc_0_0000.nii.gz',
                      'acdc': 'Task611_Biasfield_ACDC/imagesTr/patient001_frame01_loc_5_0000.nii.gz'}

# Biasf order 5
biasfield_order_5_dict = {'mm1a': 'Task905_Biasfield_MM1_A_balanced/imagesTr/A0S9V9_loc_00_0000.nii.gz',
                          'mm1b': 'Task906_Biasfield_MM1_B_balanced/imagesTr/A1D0Q7_loc_00_0000.nii.gz',
                          'acdc': 'Task613_Biasfield_ACDC/imagesTr/patient001_frame01_loc_5_0000.nii.gz'}

# Biasf order 2
biasfield_order_2_dict = {'mm1a': 'Task909_Biasfield_MM1_A_balanced/imagesTr/A0S9V9_loc_00_0000.nii.gz',
                          'mm1b': 'Task910_Biasfield_MM1_B_balanced/imagesTr/A1D0Q7_loc_00_0000.nii.gz',
                          'acdc': 'Task614_ACDC/imagesTr/patient001_frame01_loc_5_0000.nii.gz'}

# GAN sim
# Here I just copied the full paths to the images...
# Because of different name usages and such
gan_dict = {'mm1a': '/home/bme001/20184098/data/cmr7t3t/mms1_synthesis_220908/new_seven_mms1A_cut_NCE8_GAN2_220907/test_175/niftis/cmr3t2cmr7t/A0S9V9_ED1.nii.gz',
            'mm1b': '/home/bme001/20184098/data/cmr7t3t/mms1_synthesis_220908/new_seven_mms1B_cut_NCE8_GAN2_220907/test_175/niftis/cmr3t2cmr7t/A1D0Q7_ED1.nii.gz',
            'acdc': '/home/bme001/20184098/data/cmr7t3t/results/ACDC_220121/seven2three_acdc_cut_NCE4_GAN2_np128_fe_211208/test_90/niftis/cmr3t2cmr7t/patient001_frame011.nii.gz',}


dict_dict = {'No augmentation': normal_dict,
             'Simulated bias field': biasfield_sim_dict,
             '5-Polynomial bias field': biasfield_order_5_dict,
             '2-Polynomial bias field': biasfield_order_2_dict}

"""
Below we collect and load, rotate etc. all the images from all the sources
"""

all_images = []
for augm_method, dir_dict in dict_dict.items():
    logger.debug(f"\n\nStarting with data source {augm_method}")
    temp_list = []
    n_source = len(dir_dict)
    dest_file = os.path.join(DFINAL, f"example_{augm_method}.png")
    for data_source, src_file in dir_dict.items():
        src_file = os.path.join(raw_data_path, src_file)
        src_array = hmisc.load_array(src_file)
        n_slice = src_array.shape[-1]
        sel_slice = n_slice // 2
        sel_array = src_array[:, :, sel_slice]
        # Rotate stuff..
        if augm_method == 'Simulated bias field':
            sel_array = np.rot90(sel_array, k=-1)
            if data_source != 'acdc':
                sel_array = sel_array[::-1]
            else:
                sel_array = sel_array[:, ::-1]
        elif ('-Polynomial bias field' in augm_method) and (data_source != 'acdc'):
            sel_array = np.rot90(sel_array, k=-1)
            sel_array = sel_array[::-1]
        else:
            pass
        #
        temp_list.append(sel_array)
    #
    all_images.append(temp_list)
    fig_obj = hplotc.ListPlot(temp_list, col_row=(1, n_source), ax_off=True)
    hplotf.add_text_box(fig_obj.figure, 0, str(augm_method),
                        linewidth=1, position='top')
    fig_obj.figure.savefig(dest_file, bbox_inches='tight', pad_inches=0.0)


"""
Here we load the images from the GAN 
"""

temp_list = []
n_source = len(gan_dict)
dest_file = os.path.join(DFINAL, f"example_GAN.png")

for data_source, src_file in gan_dict.items():
    src_array = hmisc.load_array(src_file)
    # Logging statements
    logger.debug(f"Starting with {data_source}")
    logger.debug(f"Reading file {src_file}")
    logger.debug(f"Got array with shape {src_array.shape}")
    # /
    n_slice = src_array.shape[-2]
    sel_slice = n_slice // 2
    sel_array = np.squeeze(src_array)[:, :, sel_slice]
    temp_list.append(sel_array)

fig_obj = hplotc.ListPlot(temp_list, col_row=(1, n_source), ax_off=True)
hplotf.add_text_box(fig_obj.figure, 0, 'GAN synthesize',
                    linewidth=1, position='top')
fig_obj.figure.savefig(dest_file, bbox_inches='tight', pad_inches=0.0)

# Adding the GAN images to 'all images'
all_images.insert(2, temp_list)

"""
Here we try to smooth the image intensities of all smoothed images
"""

all_images = np.array(all_images)
logger.debug(f"\n\nThe resulting array is of shape {all_images.shape}")
n_sources = all_images.shape[1]
for i_source in range(n_sources):
    single_source_images = all_images[:, i_source]
    single_source_images = harray.scale_minmax(single_source_images, axis=(-2, -1))
    for x in single_source_images:
        logger.debug(f"Single source images max {x.max()}")
    nx = max(single_source_images.shape)
    logger.debug(f"Single source images shape {single_source_images.shape}")
    fig_obj = hplotc.ListPlot(single_source_images, ax_off=True)
    fig_obj.figure.savefig(os.path.join(DFINAL, f'single_source_img_{i_source}.png'), bbox_inches='tight', pad_inches=0.0)
    equil_obj = hplotc.ImageIntensityEqualizer(reference_image=single_source_images[0], patch_width=nx // 3,
                                               image_list=list(single_source_images[1:]), dynamic_thresholding=True,
                                               distance_measure='ssim')
    smoothed_source_images = equil_obj.correct_image_list()
    equil_obj.measure_improvement(corrected_images=smoothed_source_images)
    fig_obj = hplotc.ListPlot([single_source_images[0]] + list(smoothed_source_images), ax_off=True, col_row=(len(smoothed_source_images)+1, 1), wspace=0)
    fig_obj.figure.savefig(os.path.join(DFINAL, f'smoothed_single_source_img_{i_source}.png'),
                           bbox_inches='tight', pad_inches=0.0)

