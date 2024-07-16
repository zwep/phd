"""
Plot a bias field and synthesized image...
"""

import nibabel
import os
import data_generator.Segment7T3T as data_gen
import helper.plot_class as hplotc
import numpy as np
import imageio

ddata = '/home/bugger/Documents/presentaties/Espresso/januari_2022/biasfield_example'
gen_obj = data_gen.DataGeneratorCardiacSegment(ddata, target_type='segmentation',
                                               transform_type='abs', presentation_mode=True,
                                               dataset_type='test')

ddata_ella_b1_min = '/home/bugger/Documents/presentaties/Espresso/januari_2022/b1_distr/b1_minus/Cardiac_Array_V10_ProcessedData_4ch.npy'
ddata_ella_b1_plus = '/home/bugger/Documents/presentaties/Espresso/januari_2022/b1_distr/b1_plus/Cardiac_Array_V10_ProcessedData_4ch.npy'
A_b1min = np.abs(np.load(ddata_ella_b1_min))
A_b1plus = np.abs(np.load(ddata_ella_b1_plus))
hplotc.ListPlot([np.rot90(x, k=-1, axes=(1,2)) for x in [A_b1min]], ax_off=True, title='B1 minus')
hplotc.ListPlot([np.rot90(x, k=-1, axes=(1,2)) for x in [A_b1plus]], ax_off=True, title='B1 plus')

hplotc.ListPlot(A_b1min, ax_off=True, title='B1 minus', hspace=0, wspace=0)
hplotc.ListPlot(A_b1plus, ax_off=True, title='B1 plus', hspace=0, wspace=0)


results = []
for iimg in range(10):
    container = gen_obj.__getitem__(iimg)
    input_array = container['input']
    # rho_array = np.fliplr(np.rot90(container['target_clean'], k=-1))
    rho_array = container['target_clean']
    b1_minus_array_shimmed = container['b1m']
    b1_plus_array_shimmed = container['b1p']
    biasfield = b1_minus_array_shimmed * b1_plus_array_shimmed
    # input_array = rho_array * b1_minus_array_shimmed * b1_plus_array_shimmed
    # hplotc.ListPlot([input_array, rho_array, b1_minus_array_shimmed, b1_plus_array_shimmed], ax_off=True, augm='np.abs')
    results.append([input_array.numpy(), rho_array, biasfield])

list_input_array, list_rho_array, biasfield_array = zip(*results)
array_input = np.array(list_input_array)
array_rho = np.array(list_rho_array)
biasfield_array = np.array(biasfield_array)

hplotc.ListPlot(array_input[None, :6, 0], augm='np.abs', ax_off=True, wspace=0, hspace=0, sub_col_row=(2, 3))
hplotc.ListPlot(array_rho[None, :6], augm='np.abs', ax_off=True, wspace=0, hspace=0, sub_col_row=(2, 3))
hplotc.ListPlot(biasfield_array[None, :6], augm='np.abs', ax_off=True, wspace=0, hspace=0, sub_col_row=(2, 3))

hplotc.ListPlot([rho_array], augm='np.abs', ax_off=True)


"""
Visualization of model results..
"""

dvisual = '/home/bugger/Documents/presentaties/Espresso/januari_2022/for_performance_model_results'
dimages = '/home/bugger/Documents/presentaties/Espresso/januari_2022/7T_examples'
dlabels = '/home/bugger/Documents/presentaties/Espresso/januari_2022/7T_labels'
ddest = '/home/bugger/Documents/presentaties/Espresso/januari_2022/model_visualization'

image_files = os.listdir(dimages)
temp_images = []
temp_labels = []
for i_file in image_files:
    image_file = os.path.join(dimages, i_file)
    label_file = os.path.join(dlabels, i_file)
    image_array = np.array(nibabel.load(image_file).get_fdata())
    print(image_array.shape)
    label_array = np.array(nibabel.load(label_file).get_fdata())
    n_card = image_array.shape[-1]
    temp_images.append(image_array[:, :, n_card // 2])
    temp_labels.append(label_array[:, :, n_card // 2])


"""
Visualize nnUnet results...
"""

import helper.misc as hmisc

dnnunet = '/home/bugger/Documents/presentaties/Espresso/januari_2022/for_performance_model_results/nnunet_results'
ddest = '/home/bugger/Documents/presentaties/Espresso/januari_2022/nnunet_visualization'
nnunet_files = os.listdir(dnnunet)
for i_file in nnunet_files:
    base_name = hmisc.get_base_name(i_file)
    temp_file_path = os.path.join(dnnunet, i_file)
    orig_7T_file = os.path.join(dimages, i_file)
    nnunet_label = np.moveaxis(np.array(nibabel.load(temp_file_path).get_fdata()), -1, 0)
    image_7T = np.moveaxis(np.array(nibabel.load(orig_7T_file).get_fdata()), -1, 0)
    fig_obj = hplotc.ListPlot([nnunet_label[10:20], image_7T[10:20]], ax_off=True)
    fig_obj.figure.savefig(os.path.join(ddest, base_name + '.png'), bbox_inches='tight')


ddest_model = '/home/bugger/Documents/presentaties/Espresso/januari_2022/model_visualization'
for d, _, f in os.walk(dvisual):
    if len(f):
        print(d, f)
        temp = []
        temp_nnunet = []
        temp_image = []
        for i_file in f:
            file_path = os.path.join(d, i_file)
            real_image_path = os.path.join(dimages, i_file)
            nnunet_image_path = os.path.join(dnnunet, i_file)
            not_nnunet = np.moveaxis(np.array(nibabel.load(file_path).get_fdata()), -1, 0)[10:20]
            real_img = np.moveaxis(np.array(nibabel.load(real_image_path).get_fdata()), -1, 0)[10:20]
            nnunet = np.moveaxis(np.array(nibabel.load(nnunet_image_path).get_fdata()), -1, 0)[10:20]
            fig_obj = hplotc.ListPlot([real_img, not_nnunet, nnunet], title=os.path.basename(d), ax_off=True)

            file_name = os.path.basename(d) + "_" + hmisc.get_base_name(i_file)
            fig_obj.figure.savefig(os.path.join(ddest_model, file_name + '.png'), bbox_inches='tight')
            hplotc.close_all()

        #
        #     temp.append(not_nnunet)
        #     temp_nnunet.append(nnunet)
        #     temp_image.append(real_img)
        #
        # temp_image = np.array(temp_image)
        # temp = np.array(temp)
        # temp_nnunet = np.array(temp_nnunet)
        # fig_obj = hplotc.ListPlot([temp_image, temp, temp_nnunet], title=os.path.basename(d), ax_off=True)
        # fig_obj.figure.savefig(os.path.join(ddest, base_name + '.png'), bbox_inches='tight')
