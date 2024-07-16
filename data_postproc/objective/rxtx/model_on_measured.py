"""
Here we are going to collect all the model outputs that give the undistrubed/restored as result immediately

"""

import numpy as np
import torch
import helper.misc as hmisc
import helper.array_transf as harray
import helper.plot_class as hplotc
import objective.rx2tx.executor_rx2tx as executor
import os
import reconstruction.sensitivity_map as sens_map

"""
Prostate directories
"""

# Choose with which model we are going to generate the results
model_path_dir = '/home/bugger/Documents/model_run/rxtx_pinn' # ==>

# Paths where to store the metrics etc.
store_img_path = os.path.join(model_path_dir, 'result_images')
store_json_path = os.path.join(model_path_dir, 'result_json')
store_csv_path = os.path.join(model_path_dir, 'result_csv')

for temp_map in [store_img_path, store_json_path, store_csv_path]:
    if not os.path.isdir(temp_map):
        os.mkdir(temp_map)


main_data_path = '/media/bugger/MyBook/data/7T_data/prostate_t2_selection/'
measured_path = os.path.join(main_data_path, 't2w')
body_mask_path = os.path.join(main_data_path, 'body_mask')
prostate_mask_path = os.path.join(main_data_path, 'prostate_mask')
muscle_mask_path = os.path.join(main_data_path, 'muscle_mask')
subcutaneous_fat_mask_path = os.path.join(main_data_path, 'subcutaneous_fat_mask')

# Overview of all the models available in model_path_dir
model_path_list = [os.path.join(model_path_dir, x) for x in os.listdir(model_path_dir)]
model_path_list = [x for x in model_path_list if 'result' not in x]

body_mask_file_list = sorted([os.path.join(body_mask_path, x) for x in os.listdir(body_mask_path)])
prostate_mask_file_list = sorted([os.path.join(prostate_mask_path, x) for x in os.listdir(prostate_mask_path)])
muscle_mask_file_list = sorted([os.path.join(muscle_mask_path, x) for x in os.listdir(muscle_mask_path)])
subcutaneous_fat_mask_file_list = sorted([os.path.join(subcutaneous_fat_mask_path, x) for x in os.listdir(subcutaneous_fat_mask_path)])
# These nr of files can be more than the amount of files there are...
# SO just check against ANY mask dir, to match the files. Then sort
# This makes sure that file 0 is "the same" over all the lists
file_list = sorted([os.path.join(measured_path, x) for x in os.listdir(measured_path) if x in os.listdir(subcutaneous_fat_mask_path)])

"""
Start the full run....
"""

# Dictionaries to store metrics and images and more
overal_image_result = {}
overal_metric_result = {}


i_model_path = model_path_list[1]
for i_model_path in model_path_list:
    model_name = os.path.basename(i_model_path)
    print(i_model_path)

    config_param = hmisc.convert_remote2local_dict(i_model_path, path_prefix='/media/bugger/MyBook/data/semireal')
    # Otherwise squeeze will not work properly..
    config_param['data']['batch_size'] = 1
    config_param['model']['config_regular']['reload_weights_config']['status'] = False

    """
    Load the model
    """

    decision_obj = executor.DecisionMaker(config_file=config_param, debug=False,
                                          load_model_only=True, inference=True, device='cpu')  # ==>>
    modelrun_obj = decision_obj.decision_maker()
    modelrun_obj.load_weights()
    if modelrun_obj.model_obj:
        modelrun_obj.model_obj.eval()
    else:
        modelrun_obj.generator.eval()

    # Arrays to store results from the just loaded model
    image_result = []
    metric_result = []
    model_debug_counter = 0
    debug_plot_data = 0
    i = 1
    # Do the numbering based on the masks, they are always less
    for i in range(len(body_mask_file_list)):
        load_file = file_list[i]
        body_mask_file = body_mask_file_list[i]

        body_mask_array = np.load(body_mask_file)
        #  I think it is unfair to use t2w images....
        # Lets try a survey...

        ddata = '/media/bugger/MyBook/data/7T_scan/prostate/2021_01_06/pr_16289/pr_06012021_1635397_5_2_surveyisoV4.cpx'
        import reconstruction.ReadCpx as read_cpx
        cpx_obj = read_cpx.ReadCpx(ddata)
        input_cpx = cpx_obj.get_cpx_img()
        input_cpx = np.rot90(np.squeeze(input_cpx[:, 1]), axes=(-2, -1))
        hplotc.ListPlot([input_cpx], augm='np.abs')
        input_sum_of_absolutes = np.abs(input_cpx).sum(axis=0)
        body_mask_array = harray.get_treshold_label_mask(input_sum_of_absolutes).astype(float)
        # hplotc.ListPlot([abs_sum, abs_mask], augm='np.abs')
        # #
        input_cpx = np.load(load_file)

        input_cpx = harray.scale_minpercentile_both(input_cpx, is_complex=True, q=98)
        input_cpx = harray.scale_minmax(input_cpx, is_complex=True)

        input_sum_of_absolutes = np.abs(input_cpx).sum(axis=0)

        # # # /Espirit approach on t2w data.. or survey data... create A_tensor.
        import reconstruction.ReadCpx as read_cpx

        # cpx_obj = read_cpx.ReadCpx('/media/bugger/MyBook/data/7T_scan/prostate/2021_02_11/V9_17048/v9_11022021_1627064_1_3_surveyisoV4.cpx')

        input_cpx = np.squeeze(cpx_obj.get_cpx_img())
        input_cpx = np.sum(input_cpx, axis=0)
        # input_cpx = np.squeeze(cpx_obj.get_cpx_img()[:, 1])
        input_cpx = np.rot90(input_cpx, axes=(-2, -1))
        input_cpx = input_cpx[-8:]
        hplotc.ListPlot([input_cpx], augm='np.abs', ax_off=True, start_square_level=2)
        hplotc.ListPlot([np.abs(input_cpx).sum(axis=0)], augm='np.abs', ax_off=True, start_square_level=2)

        kspace_array = harray.transform_image_to_kspace_fftn(input_cpx, dim=(-2, -1))
        espirit_obj = sens_map.EspiritCalib(kspace_array)
        b1m_array_espirit = espirit_obj.run()
        b1m_array_espirit = harray.scale_minmax(b1m_array_espirit, is_complex=True)
        body_mask_obj = hplotc.MaskCreator(b1m_array_espirit)
        body_mask_array = body_mask_obj.mask
        # hplotc.ListPlot(b1m_array_espirit * body_mask_array, augm='np.abs')
        x_input = b1m_array_espirit * body_mask_array[None]
        x_input = harray.correct_mask_value(x_input, body_mask_array)
        hplotc.ListPlot(x_input, augm='np.angle')
        n_y, n_x = x_input.shape[-2:]
        x_inputed = harray.to_stacked(x_input, cpx_type='cartesian', stack_ax=0)
        x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
        # new_size = tuple((np.array(x_inputed.shape[-2:]) * 0.5).astype(int))
        # import skimage.transform
        # x_inputed = skimage.transform.resize(x_inputed, (16, ) + new_size, preserve_range=True)
        A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()
        # /Espirit

        # B1 shim serie - SVD
        d = '/home/bugger/Documents/data/7T/prostate/prostate_7t/2021_01_06/pr_16289'
        b1_shim_file = 'pr_06012021_1633352_4_2_b1shimseriesV4'
        # cpx_obj_shim = read_cpx.ReadCpx(os.path.join(d, b1_shim_file))
        cpx_obj_shim = read_cpx.ReadCpx('/media/bugger/B7DF-4571/2022_06_15/V9_36792/v9_15062022_1641373_3_3_b1shimseriesV4.cpx')

        cpx_shim_array = np.squeeze(cpx_obj_shim.get_cpx_img())
        cpx_shim_array = np.rot90(cpx_shim_array, axes=(-2, -1))

        hplotc.ListPlot(np.abs(cpx_shim_array))

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
        global_array = cpx_shim_array

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

        # THis took a really long time on my CPU...
        eig_list, left_list, right_list = zip(*results)
        for i, i_iter in enumerate(cross_prod):
            sel_y, sel_x = i_iter
            left_svd_array[:, sel_y, sel_x] = left_list[i]
            right_svd_array[:, sel_y, sel_x] = right_list[i]
            svd_array[sel_y, sel_x] = eig_list[i]

        hplotc.ListPlot(left_svd_array, augm='np.abs')
        x_input = left_svd_array
        n_y, n_x = x_input.shape[-2:]
        x_inputed = harray.to_stacked(x_input, cpx_type='cartesian', stack_ax=0)
        x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
        # new_size = tuple((np.array(x_inputed.shape[-2:]) * 0.5).astype(int))
        # import skimage.transform
        # x_inputed = skimage.transform.resize(x_inputed, (16, ) + new_size, preserve_range=True)
        A_tensor = torch.as_tensor(x_inputed[np.newaxis]).float()
        # /B1 shim serie - SVD

        # Model
        # x_input = input_cpx * body_mask_array[None]
        x_input = input_cpx * np.exp(-1j * np.angle(input_cpx[0]))
        n_y, n_x = x_input.shape[-2:]
        x_inputed = harray.to_stacked(x_input, cpx_type='cartesian', stack_ax=0)
        x_inputed = x_inputed.T.reshape((n_x, n_y, -1)).T
        # Check if resizing makes any difference in the output...
        import skimage.transform
        x_inputed = skimage.transform.resize(x_inputed, (16, 256, 256), preserve_range=True)
        body_mask_array = skimage.transform.resize(body_mask_array, (256, 256), preserve_range=True)
        hplotc.ListPlot(body_mask_array)
        A_tensor = torch.as_tensor(x_inputed[np.newaxis]*body_mask_array).float()

        # Use A_tensor in the model..
        with torch.no_grad():
            if modelrun_obj.config_param['model']['model_choice'] == 'gan':
                output = modelrun_obj.generator(A_tensor)
            elif modelrun_obj.config_param['model']['model_choice'] == 'cyclegan':
                output = modelrun_obj.netG_A2B(A_tensor)
            else:
                try:
                    output = modelrun_obj.model_obj(A_tensor)
                except RuntimeError as e:
                    nx, ny = A_tensor.shape[-2:]
                    # ???
                    new_ny = int(ny / 2 ** 5) * 2 ** 5
                    new_nx = int(nx / 2 ** 5) * 2 ** 5
                    print(nx, ny, "  ", new_nx, new_ny)
                    body_mask_array = body_mask_array[:new_nx, :new_ny]
                    output = modelrun_obj.model_obj(A_tensor[:, :, :new_nx, :new_ny])

        output_cpx = harray.to_complex_chan(np.moveaxis(np.array(output)[0], 0, -1), img_shape=output.shape[-2:])
        hplotc.ListPlot([x_input * body_mask_array], augm='np.angle', start_square_level=2, cbar=True, ax_off=True)
        hplotc.ListPlot([output_cpx[0] * body_mask_array], augm='np.angle', start_square_level=2, cbar=True, ax_off=True)
        hplotc.ListPlot([output_cpx[0] * body_mask_array], augm='np.abs', start_square_level=2, cbar=True, ax_off=True)
        # hplotc.ListPlot([output_cpx[0] * body_mask_array], augm='np.abs', start_square_level=2)