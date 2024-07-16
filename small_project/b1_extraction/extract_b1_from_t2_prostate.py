"""
Here we tried to get the B1+ field from the acquisition... using our predicted bias field or rho field...

However.. after an afternoon of trying
I did not succeed.

The obtain B1+ field still looked ALOT like the bias field itself. Hence, the extracted B1- field
changed too little. Here we used the following relationship
Biasfield = B1- * f(B1+)

where f(.) is an unknown function, assumed to be sin(x) for now.

We had issues with noise and with scaling. I dont know how to scale each image
"""
import helper.array_transf
import objective.inhomog_removal.executor_inhomog_removal as executor
import objective.inhomog_removal.postproc_inhomog_removal as postproc_inhomog
import helper.plot_class as hplotc
import os

# needed these for some misc coding
import helper.array_transf as harray
import matplotlib.pyplot as plt
import numpy as np
import helper.misc as hmisc
import pydicom
import os

main_dir = '/home/bugger/Documents/data/7T/prostate'
# '/media/bugger/MyBook/data/7T_data
input_dir = os.path.join(main_dir, '/prostate_t2_selection/t2w_n4itk/input')
input_dir_multi = os.path.join(main_dir, 'prostate_t2_selection/t2w')
mask_dir = os.path.join(main_dir, 'prostate_t2_selection/body_mask')
base_model_dir = '/home/bugger/Documents/model_run'
both_single_config_path = os.path.join(base_model_dir, 'inhomog_single_both/resnet_2022_06')
both_multi_config_path = os.path.join(base_model_dir, 'inhomog_multi_both/resnet_2022_06')

# These are the dicts for patient evaluation.
both_single_dict_volunteer = {"dconfig": both_single_config_path, "dimage": input_dir, "dmask": mask_dir,
                     "ddest": both_single_config_path}
# These are the dicts for patient evaluation.
both_multi_dict_volunteer = {"dconfig": both_multi_config_path, "dimage": input_dir_multi, "dmask": mask_dir,
                     "ddest": both_multi_config_path}

dict_list = [both_multi_dict_volunteer]

for temp_dict in dict_list:
    base_name = os.path.basename(temp_dict['ddest'])
    model_name = os.path.basename(os.path.dirname(temp_dict['ddest']))
    stride = 64
    mask_ext = '.npy'
    mask_suffix = ''
    target_dir = temp_dict.get('dtarget', None)
    postproc_obj = postproc_inhomog.PostProcInhomogRemoval(image_dir=temp_dict['dimage'],
                                                           mask_dir=temp_dict['dmask'],
                                                           dest_dir=temp_dict['ddest'],
                                                           target_dir=target_dir,
                                                           config_path=temp_dict['dconfig'],
                                                           executor_module=executor, config_name='config_param.json',
                                                           stride=stride, patch_shape=(256, 256),
                                                           storage_extension='nii',
                                                           mask_ext=mask_ext,
                                                           mask_suffix=mask_suffix)


    postproc_obj.load_file(0)
    res = postproc_obj.run_slice_patched(0)
    pred_img, pred_biasf = res
    hplotc.ListPlot([pred_img, pred_biasf, postproc_obj.sum_of_absolute_img])

    # Now lets get the B1-minus fields..
    # We need the complex data for that.. because we need the k-space data.
    import reconstruction.ReadCpx as read_cpx
    import reconstruction.sensitivity_map as sens_map
    selected_file_name = hmisc.get_base_name(postproc_obj.file_list[0])
    # ddata_scan = '/media/bugger/MyBook/data/7T_scan/prostate'
    ddata_scan = '/home/bugger/Documents/data/7T/prostate/prostate_7t'
    found_directory = None
    for d, _, f in os.walk(ddata_scan):
        filter_f = [x for x in f if selected_file_name in x and x.endswith('cpx')]
        # filter_f_b1_shim = [x for x in f if selected_file_name in x and x.endswith('cpx')]
        if len(filter_f):
            print("Loading cpx object ", d, filter_f[0])
            cpx_file = filter_f[0]
            cpx_obj_array = read_cpx.ReadCpx(os.path.join(d, cpx_file))
            b1_shim_file = [x for x in f if 'b1shim' in x and x.endswith('cpx')][-1]
            cpx_obj_shim = read_cpx.ReadCpx(os.path.join(d, b1_shim_file))
            # Not getting the B1-map here.. because I dont know how to process that one in the .cpx format
            # I do have it in DICOM format. That should be easier.

    cpx_array = np.squeeze(cpx_obj_array.get_cpx_img())
    cpx_array = np.rot90(cpx_array, axes=(-2, -1))

    cpx_shim_array = np.squeeze(cpx_obj_shim.get_cpx_img())
    cpx_shim_array = np.rot90(cpx_shim_array, axes=(-2, -1))
    # This shift is needed, otherwise a part of the image is on the wrong side...
    hplotc.ListPlot(np.abs(cpx_shim_array).sum(axis=0).sum(axis=0), title='shim array')
    # cpx_shim_array = np.roll(cpx_shim_array, 25, axis=3)
    # Display the cpx array and shim array..
    hplotc.ListPlot([np.abs(cpx_array).sum(axis=0), np.abs(cpx_shim_array).sum(axis=0).sum(axis=0)], augm='np.abs',
                    title='t2w array and shim array - check allignment')
    # # # We need to register this cpx shim array to the cpx array...
    # They are off... both in size and position
    cpx_shim_mask = harray.get_treshold_label_mask(np.abs(cpx_shim_array).sum(axis=0).sum(axis=0))
    tresh_value = 0.58*np.mean(np.abs(cpx_array).sum(axis=0))
    cpx_array_mask = harray.get_treshold_label_mask(np.abs(cpx_array).sum(axis=0), treshold_value=tresh_value)
    hplotc.ListPlot([cpx_shim_mask, cpx_array_mask], title='check mask creation')
    # Now do a center-crop
    _, cpx_shim_crop_coords = harray.get_center_transformation_coords(cpx_shim_mask)
    cpx_shim_mask_cropped = harray.apply_crop(cpx_shim_mask, crop_coords=cpx_shim_crop_coords)
    _, cpx_array_crop_coords = harray.get_center_transformation_coords(cpx_array_mask)
    cpx_array_mask_cropped = harray.apply_crop(cpx_array_mask, crop_coords=cpx_array_crop_coords)

    # Cropped CPX array:
    cpx_array_cropped = np.array([harray.apply_crop(x, crop_coords=cpx_array_crop_coords) for x in cpx_array])
    cpx_shim_array_cropped = np.array([[harray.apply_crop(y, crop_coords=cpx_shim_crop_coords)for y in x] for x in cpx_shim_array])
    pred_img_cropped = harray.apply_crop(pred_img, crop_coords=cpx_array_crop_coords)
    pred_biasf_cropped = harray.apply_crop(pred_biasf, crop_coords=cpx_array_crop_coords)

    # # # Test for B1 minus extraction
    # I could parralelize this...
    def calc_svd(args):
        global global_array
        sel_y, sel_x = args
        sel_array = np.take(global_array, sel_x, axis=-1)
        sel_array = np.take(sel_array, sel_y, axis=-1)
        left_x, eig_x, right_x = np.linalg.svd(sel_array, full_matrices=False)
        right_x = right_x.conjugate().T
        return eig_x[0], left_x[:, 0], right_x[:, 0]

    import multiprocessing as mp
    import itertools
    N = mp.cpu_count()
    global_array = cpx_shim_array_cropped

    _, n_c, im_y, im_x = global_array.shape
    n_svd = 1
    left_svd_array = np.empty((n_c, im_y, im_x), dtype=complex)
    svd_array = np.empty((im_y, im_x), dtype=complex)
    right_svd_array = np.empty((n_c, im_y, im_x), dtype=complex)

    cross_prod = list(itertools.product(range(im_y), range(im_x)))
    print('Amount of CPUs ', N)
    print('Amount of iterations ', im_y * im_x)
    with mp.Pool(processes=N) as p:
        results = p.map(calc_svd, list(cross_prod))

    eig_list, left_list, right_list = zip(*results)
    for i, i_iter in enumerate(cross_prod):
        sel_y, sel_x = i_iter
        left_svd_array[:, sel_y, sel_x] = left_list[i]
        right_svd_array[:, sel_y, sel_x] = right_list[i]
        svd_array[sel_y, sel_x] = eig_list[i]

    hplotc.ListPlot(left_svd_array, augm='np.abs')
    # abs_left_svd_array = np.abs(left_svd_array)
    abs_left_svd_array = np.abs(left_svd_array)
    import skimage.transform as sktransf
    b1m_array_svd = sktransf.resize(abs_left_svd_array, cpx_array_cropped.shape, anti_aliasing=False)
    summed_b1_minus_svd = np.sqrt(b1m_array_svd.mean(axis=0))
    # b1m_array_svd = harray.resize_complex_array(left_svd_array, cpx_array_cropped.shape)
    """# # # Try to get the B1minus with ESPIRIT
    - not succesful"""
    # kspace_cpx_array = harray.transform_image_to_kspace_fftn(cpx_array)
    # espirit_obj = sens_map.EspiritCalib(kspace_cpx_array)
    # b1m_array_espirit = espirit_obj.run()
    # This is not the best b1- map...
    # hplotc.ListPlot([b1m_array_espirit, pred_biasf[None]/b1m_array_espirit], augm='np.abs')
    #
    measured_signal = np.abs(cpx_array_cropped)
    measured_signal = harray.scale_minmax(measured_signal)
    # Divide out the rho density...
    biasfield_per_coil = helper.array_transf.correct_inf_nan(measured_signal / pred_img_cropped)
    biasfield_per_coil = harray.treshold_percentile_both(biasfield_per_coil, q=99)
    hplotc.ListPlot(biasfield_per_coil)
    # First way to calc b1 plus per coil....
    calc_b1_plus_per_coil_1 = biasfield_per_coil/summed_b1_minus_svd
    calc_b1_plus_per_coil_1 = helper.array_transf.correct_inf_nan(calc_b1_plus_per_coil_1)
    calc_b1_plus_per_coil_1 = harray.treshold_percentile_both(calc_b1_plus_per_coil_1, q=99)
    calc_b1_plus_per_coil_1 = harray.scale_minmax(calc_b1_plus_per_coil_1)
    hplotc.ListPlot(calc_b1_plus_per_coil_1)
    hplotc.ListPlot(np.abs(calc_b1_plus_per_coil_1).sum(axis=0))
    # Second way to calc b1 plus per coil....

    calc_b1_plus_per_coil_2 = harray.scale_minmax(np.abs(cpx_array_cropped)) / (pred_img_cropped)
    calc_b1_plus_per_coil_2 = helper.array_transf.correct_inf_nan(calc_b1_plus_per_coil_2)
    calc_b1_plus_per_coil_2 = harray.treshold_percentile_both(calc_b1_plus_per_coil_2, q=99)
    # hplotc.ListPlot(calc_b1_plus_per_coil_2 / summed_b1_minus_svd)
    calc_b1_plus_per_coil_2 = calc_b1_plus_per_coil_2 / summed_b1_minus_svd
    calc_b1_plus_per_coil_2 = harray.scale_minmax(calc_b1_plus_per_coil_2)
    hplotc.ListPlot(calc_b1_plus_per_coil_2, vmin=(0,1))

    center_mask_b1plus = harray.create_random_center_mask(calc_b1_plus_per_coil_2.shape, random=False)
    calc_b1_plus_per_coil_2_scaled = calc_b1_plus_per_coil_2 / calc_b1_plus_per_coil_2[center_mask_b1plus == 1].mean()
    hplotc.ListPlot(calc_b1_plus_per_coil_2_scaled, title='inspect the f(b1+) map')
    plt.hist(calc_b1_plus_per_coil_2_scaled.ravel())
    b1_summed = np.arcsin((calc_b1_plus_per_coil_2) ** (1/3))
    # b1_summed = np.arcsin(harray.scale_minmax(pred_biasf_cropped))
    hplotc.ListPlot(b1_summed)

    # Get the true B1map..
    # Still not sure how to fix this...
    input_dir_dicom = '/media/bugger/MyBook/data/7T_scan/prostate_dicom/2021_01_06/pr_16289'
    b1_dicom_dir = [os.path.join(input_dir_dicom, x) for x in os.listdir(input_dir_dicom) if 'B1map' in x]
    b1_dicom_files = [os.path.join(x, 'DICOM/IM_0002') for x in b1_dicom_dir]
    b1_dicom_obj = [pydicom.read_file(x) for x in b1_dicom_files]
    b1_dicom_array = np.array([x.pixel_array for x in b1_dicom_obj])
    # hplotc.SlidingPlot(b1_dicom_array)
    # Take the first file..
    # That is.. print(b1_dicom_files[0])
    b1_rho_sel = b1_dicom_array[0, 0]
    b1_rho_mask = harray.get_treshold_label_mask(b1_rho_sel)
    b1_sel = b1_dicom_array[0, 2]
    _, b1_crop_coords = harray.get_center_transformation_coords(b1_rho_mask)
    b1_cropped = harray.apply_crop(b1_sel * b1_rho_mask, crop_coords=b1_crop_coords)
    b1_rho_mask_croped = harray.apply_crop(b1_rho_mask, crop_coords=b1_crop_coords)
    b1_cropped = sktransf.resize(b1_cropped, calc_b1_plus_per_coil_2.shape, preserve_range=True, anti_aliasing=False)
    b1_rho_mask_croped = sktransf.resize(b1_rho_mask_croped, calc_b1_plus_per_coil_2.shape, preserve_range=True, anti_aliasing=False)

    center_mask_b1_cropped = harray.create_random_center_mask(b1_cropped.shape, random=False)
    b1_cropped_scaled = b1_cropped / b1_cropped[center_mask_b1_cropped == 1].mean()
    hplotc.ListPlot([b1_cropped, pred_biasf_cropped], subtitle=[['B1'], ['predicted biasfield']])
    # b1_cropped_scaled[np.isclose(b1_cropped_scaled, 0)] = 0
    # calc_b1_plus_per_coil_2_scaled[np.isclose(calc_b1_plus_per_coil_2_scaled, 0)] = 0
    plt.scatter((harray.scale_minmax(b1_cropped_scaled)), np.arcsin(harray.scale_minmax(pred_biasf_cropped)), alpha=0.01)
    plt.plot(np.arcsin(np.sin(np.arange(0, np.pi, 0.1))))
    # plt.scatter(b1_cropped_scaled[b1_rho_mask_croped==1].ravel(), calc_b1_plus_per_coil_2_scaled[b1_rho_mask_croped==1].ravel(), alpha=0.01)
    plt.plot(np.sin(np.deg2rad(np.arange(0, 180))), 'k')
    plt.plot(np.sin(np.deg2rad(np.arange(0, 180))) ** 3, 'r')

    hplotc.ListPlot(b1_cropped_scaled)
    fig, ax = plt.subplots()
    ax.scatter(b1_cropped_scaled.ravel(), calc_b1_plus_per_coil_2_scaled.ravel(), alpha=0.01)
    ax.set_xlabel('B1 - values')
    ax.set_ylabel('Calculated B1 - values')

    # Why cant I invert this relation..?
    import numpy as np
    import matplotlib.pyplot as plt
    x_range = np.arange(0, np.pi, 0.01)
    X, Y = np.meshgrid(x_range, x_range)
    R = np.sqrt(X** 2 + Y**2)
    Z = np.sin(X+Y)
    plt.imshow(np.arcsin(Z))
    plt.scatter(R.ravel(), Z.ravel(), alpha=0.01)
    plt.scatter(R.ravel(), np.arcsin(Z).ravel(), alpha=0.01)