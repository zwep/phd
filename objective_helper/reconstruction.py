import getpass
if getpass.getuser() == 'sharreve':
    import matplotlib
    matplotlib.use('Agg')

from model.FBPConvnet import FBPCONVNet
import skimage.transform as sktransf
from skimage.metrics import structural_similarity
import helper.array_transf as harray
from objective_configuration.reconstruction import DRESULT, DDATA, ANATOMY_LIST, PERCENTAGE_LIST, \
    DINFERENCE, DRESULT_INFERENCE, FONTSIZE_XTICKS, FONTSIZE_YTICKS, DRETRO, DRESULT_RETRO, FONTSIZE_TITLE
import torch
import sigpy
import sigpy.mri
from skimage.metrics import peak_signal_noise_ratio as psnr
import re
import pandas as pd
import scipy.io
import copy
import reconstruction.ReadCpx as read_cpx
import numpy as np
import small_project.homogeneity_measure.metric_implementations as homog_metric
import helper.misc as hmisc
import helper.plot_class as hplotc
import os
from loguru import logger
import small_project.homogeneity_measure.metric_implementations as homog_measure
from helper.flip_api import compute_ldrflip

import matplotlib.pyplot as plt
import helper.metric as hmetric
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

"""
This should contain helpers specific for the reconstruction project

"""



class CollectConvergence:
    def __init__(self, experiment_path, debug=False):
        self.debug = debug
        # experiment path contains .pt files etc.
        experiment_name = os.path.basename(experiment_path)
        self.experiment_path = experiment_path
        self.dest_dir = os.path.dirname(experiment_path)
        self.dest_plot = os.path.join(self.dest_dir, experiment_name + '_validation_metric.png')

        self.file_ext = 'json'
        file_list = os.listdir(self.experiment_path)
        self.val_5x = [x for x in file_list if x.startswith('metrics_val_5x') and x.endswith(self.file_ext)]
        self.val_10x = [x for x in file_list if x.startswith('metrics_val_10x') and x.endswith(self.file_ext)]

        self.re_iteration_nr = re.compile('([0-9]+)\.json')

        if self.debug:
            logger.debug('Defined the 5x and 10x validation metrics')
            logger.debug(f'5x file length {len(self.val_5x)}' )
            logger.debug(f'10x file length {len(self.val_10x)}')


class VisualizePercentageConvergence:
    """
    This visualizes the convergences over the different percentages (using the last iteration)
    """
    def __init__(self, model_path, debug=False):
        # Here model_path points to something that contains 25p, 50p, 75p, and 100p
        # First find the suitable directories that contain .metirc json files
        self.debug = debug
        self.model_path = model_path
        self.metric_5x_dict = self.collect_metric_dict('5x')
        self.metric_10x_dict = self.collect_metric_dict('10x')

    def collect_metric_dict(self, type='5x'):
        collected_metric_dict = []
        for i_percentage in PERCENTAGE_LIST:
            # Default average metric
            average_metric_dict = {'calgary_campinas_psnr_metric': 0,
                                   'calgary_campinas_ssim_metric': 0,
                                   'calgary_campinas_vif_metric': 0}
            percentage_model_path = os.path.join(self.model_path, str(i_percentage) + 'p', 'train_mixed')
            if os.path.isdir(percentage_model_path):
                temp_collect_obj = CollectConvergence(percentage_model_path)
                if len(temp_collect_obj.val_5x) and len(temp_collect_obj.val_10x):
                    if type == '5x':
                        last_file = temp_collect_obj.val_5x[-1]
                    else:
                        last_file = temp_collect_obj.val_10x[-1]
                    last_file = os.path.join(percentage_model_path, last_file)
                    temp_json = hmisc.load_json(last_file)
                    average_metric = pd.DataFrame.from_dict(temp_json).mean(axis=1)
                    average_metric_dict = dict(average_metric)
                else:
                    pass
            else:
                pass

            collected_metric_dict.append(average_metric_dict)

        metric_dict = hmisc.listdict2dictlist(collected_metric_dict)
        return metric_dict

    def visualize_metrics(self):
        fig, ax = plt.subplots(2, figsize=(10, 10))
        title_ax0 = '5x acceleration validation metrics'
        title_ax1 = '10x acceleration validation metrics'
        plot_on_axis(x_value=PERCENTAGE_LIST, y_dict=self.metric_5x_dict, ax=ax[0], title=title_ax0, xlabel='Percentage')
        plot_on_axis(x_value=PERCENTAGE_LIST, y_dict=self.metric_10x_dict, ax=ax[1], title=title_ax1, xlabel='Percentage')
        fig.subplots_adjust(hspace=0, wspace=0.5)
        plt.tight_layout()
        return fig


class VisualizeConvergence(CollectConvergence):
    """
    This visualizes the convergences over the iterations of a given model training
    """
    def __init__(self, experiment_path, debug=False):
        super().__init__(experiment_path=experiment_path, debug=debug)

        # Execute only if we have both file types
        self.fig = None
        if len([x for x in self.val_5x if x.endswith(self.file_ext)]) and len([x for x in self.val_10x if x.endswith(self.file_ext)]):
            self.fig = self.visualize_metrics()

    def extract_iteration_number(self, x):
        re_find = self.re_iteration_nr.findall(x)
        if len(re_find):
            iteration_number = int(re_find[0])
        else:
            iteration_number = None
        return iteration_number

    def visualize_metrics(self):
        iteration_val_5x, metric_val_5x = self.process_metrics(self.val_5x)
        iteration_val_10x, metric_val_10x = self.process_metrics(self.val_10x)
        fig, ax = plt.subplots(2, figsize=(10, 10))
        title_ax0 = '5x acceleration validation metrics'
        title_ax1 = '10x acceleration validation metrics'
        plot_on_axis(x_value=iteration_val_5x, y_dict=metric_val_5x, ax=ax[0], title=title_ax0)
        plot_on_axis(x_value=iteration_val_10x, y_dict=metric_val_10x, ax=ax[1], title=title_ax1)
        # fig.legend()
        fig.subplots_adjust(hspace=0, wspace=0.5)
        plt.tight_layout()
        return fig

    def process_metrics(self, file_list):
        sorted_list = sorted(file_list, key=lambda x: self.extract_iteration_number(x))
        metric_list = []
        iteration_number_list = []
        for i_file in sorted_list:
            if self.debug:
                logger.debug(f'Processing file {i_file}', end='\r')
            json_path = os.path.join(self.experiment_path, i_file)
            iteration_number = self.extract_iteration_number(i_file)
            temp_json = hmisc.load_json(json_path)
            average_metric = pd.DataFrame.from_dict(temp_json).mean(axis=1)
            average_metric_dict = dict(average_metric)
            #
            if self.debug:
                logger.debug(f'Average mean metrics {average_metric}')
            metric_list.append(average_metric_dict)
            iteration_number_list.append(iteration_number)
        # Convert the list of dictionaries to a dictionary of lists
        if self.debug:
            logger.debug('Processed the list of metrics')
        metric_dict = hmisc.listdict2dictlist(metric_list)
        return iteration_number_list, metric_dict

    def savefig(self):
        assert self.fig is not None
        logger.debug(f'Storing figure at {self.dest_plot}')
        self.fig.savefig(self.dest_plot, dpi=300, bbox_inches='tight', pad_inches=0.0)

    def write_optimal_model_pt_file(self):
        # This is done based on 5x metrics and SSIM
        iteration_val_5x, metric_val_5x = self.process_metrics(self.val_5x)
        sorted_diff = np.argsort(np.diff(metric_val_5x['calgary_campinas_ssim_metric']))
        # If we have found something... sort on index
        # Get the nr. of iterations that belong to that
        if len(sorted_diff):
            ind_largest_diff = sorted_diff[-1]
            model_file_nr = iteration_val_5x[ind_largest_diff + 2]
        else:
            # Or just take the last nr of iterations
            model_file_nr = iteration_val_5x[-1]

        with open(os.path.join(self.experiment_path, 'model_number'), 'w') as f:
            f.write(str(model_file_nr))
        return model_file_nr


class FileGatherer:
    """
    Mother class to make stuff easier

    I could refactor this to helper/reconstruction...
    Some things get a bit diluted...

    I have
        objective_helper
            reconstruction.py
    But also..
        helper
            reconstruction.py

    There are more cases like this...

    """
    def __init__(self, pred_folder, ext='h5', inference_bool=False, retro_bool=False, debug=False, calc_us=True):
        self.ext = ext
        self.debug = debug
        self.calc_us = calc_us

        # We split this into inference and evaluation...
        # For inference we need to find the target array in a different way
        # Also.. we dont need an undersampled version of the input in that case
        if inference_bool:
            self.data_path = DINFERENCE
            self.data_pred = os.path.join(DRESULT_INFERENCE, pred_folder)

            # Get prediction files
            self.file_list_pred, self.pred_array = self.load_pred_array()
            # Get target and undersampled array (input)
            self.target_array, self.us_array = self.load_fully_sampled_files(self.file_list_pred)
        elif retro_bool:
            self.data_path = DRETRO
            self.data_pred = os.path.join(DRESULT_RETRO, pred_folder)

            # Get prediction files
            self.file_list_pred, self.pred_array = self.load_pred_array()
            # Hack into this
            self.target_array = self.us_array = self.pred_array
        else:
            self.data_path = DDATA
            self.data_pred = os.path.join(DRESULT, pred_folder)

            # Get prediction files
            self.file_list_pred, self.pred_array = self.load_pred_array()

            # Find the original files that were used for prediction
            file_list_input = self.get_original_files(self.file_list_pred)

            # Re-create the 5x and 10x undersampling images
            self.target_array, self.us_array = self.get_undersampled_img(file_list_input)

    def load_pred_array(self):
        file_list_pred = [x for x in os.listdir(self.data_pred) if x.endswith(self.ext)]
        #
        pred_array = []
        for i_file in file_list_pred:
            sel_path = os.path.join(self.data_pred, i_file)
            temp_array = hmisc.load_array(sel_path, data_key='reconstruction')
            pred_array.append(temp_array)

        logger.debug(f'Number of prediction files {len(file_list_pred)}')
        logger.debug(f'Shape of prediction array {[x.shape for x in pred_array]}')
        return file_list_pred, pred_array

    def load_fully_sampled_files(self, file_list):
        under_sampled_array = []
        fully_sampled_array = []
        for ii, i_file in enumerate(file_list):
            # base_name = hmisc.get_base_name(i_file)
            input_file_path = os.path.join(self.data_path, 'input', i_file)
            target_file_path = os.path.join(self.data_path, 'target', i_file)
            logger.debug(f'\nLoading file {input_file_path}')
            logger.debug(f'Loading file {target_file_path}')
            input_array = hmisc.load_array(input_file_path, data_key='kspace')
            target_array = hmisc.load_array(target_file_path, data_key='kspace')

            input_array = convert_to_sos(input_array)
            target_array = convert_to_sos(target_array)
            logger.debug(f'Shapes... {input_array.shape} {target_array.shape}')
            # This makes sure that we have an equal sized input and target array (and pred array)
            # n_card_us = input_array.shape[0]
            # n_card_fs = target_array.shape[0]
            # logger.debug(f'Number of cardiac phases {n_card_us} / {n_card_fs}')
            # n_repeat = np.ceil(n_card_us / n_card_fs)
            # # Repeat it..
            # sos_repeat_array = np.repeat(target_array, n_repeat, axis=0)
            # # But make sure we only take at max ncardfs examples
            # # This might become ugly.. but it is practical at this momoent
            # sos_repeat_array = sos_repeat_array[:n_card_us]
            # We are not taking all the cardiac phases anymore.
            # I am just not sure about the alignment.
            # input_array = input_array[:1]
            # target_array = target_array[:1]
            under_sampled_array.append(input_array)
            # fully_sampled_array.append(sos_repeat_array)
            fully_sampled_array.append(target_array)
        return fully_sampled_array, under_sampled_array

    def get_original_files(self, file_list_pred):
        file_list_input = []
        for i_pred_file in file_list_pred:
            # First find the original location of the pred files
            for d, _, f in os.walk(self.data_path):
                # Make sure that we dont also take from 'mixed'
                # There are no other anatomies besides mixed
                # Also... if we are dealing with inference, then there is a different target file
                if (i_pred_file in f):  # and (any([x in d for x in ANATOMY_LIST])):
                    input_file = os.path.join(d, i_pred_file)
                    file_list_input.append(input_file)
                    break
        #
        logger.debug(f'Number of input files {len(file_list_input)}')
        return file_list_input

    def get_undersampled_img(self, file_list_input):
        input_list_sos_us = []
        input_list_sos = []
        acceleration = 5
        if '10x' in self.data_pred:
            acceleration = 10
        #
        logger.debug(f'Acceleration factor {acceleration}')
        #
        for i_file in file_list_input:
            sel_path = os.path.join(self.data_path, i_file)
            temp_array = hmisc.load_array(sel_path, data_key='kspace')
            temp_array_cpx = temp_array[..., ::2] + 1j * temp_array[..., 1::2]
            # Now it is (loc, coil, ny nx)
            temp_array_cpx = np.moveaxis(temp_array_cpx, -1, 1)
            temp_array_img = np.fft.ifft2(np.fft.fftshift(temp_array_cpx, axes=(-2, -1)))
            sos = np.sqrt(np.sum(np.abs(temp_array_img) ** 2, axis=1))

            if self.calc_us:
                nx, ny = temp_array_img.shape[-2:]
                traj = undersample_trajectory((nx, ny), n_points=max(nx, ny), p_undersample=100//acceleration)

                # Should do this for each location actually..
                temp_us = np.array([undersample_img(x, traj=traj) for x in temp_array_img])
                # temp_us = undersample_img_CIRCUS(temp_array_img, acceleration=acceleration)
                sos_us = np.sqrt(np.sum(np.abs(temp_us) ** 2, axis=1))
            else:
                sos_us = np.copy(sos)

            input_list_sos_us.append(sos_us)
            input_list_sos.append(sos)
        # Used to make an array out of this..
        # But that created a ragged list...
        input_us = input_list_sos_us
        input_array = input_list_sos
        logger.debug(f'Number of SoS US images {len(input_list_sos_us)}')
        logger.debug(f'Example shape of input u.s. array {input_us[0].shape}')
        logger.debug(f'Example shape of input array {input_array[0].shape}')
        return input_array, input_us


class CalculateMetric(FileGatherer):
    def __init__(self, pred_folder, ext='h5', ddest=None, inference_bool=False, calc_us=True):
        super().__init__(pred_folder=pred_folder, ext=ext, inference_bool=inference_bool, calc_us=calc_us)
        if ddest is None:
            self.ddest = self.data_pred
        else:
            self.ddest = ddest

    def calculate(self):
        n_img = len(self.pred_array)
        metric_list = []
        for i_img in range(n_img):
            n_slice_pred = self.pred_array[i_img].shape[0]
            n_slice_target = self.target_array[i_img].shape[0]
            if n_slice_pred != n_slice_target:
                print('WARNING THE SLICES ARE NOT EQUAL... MISALIGNEMNT')
                print(n_slice_pred, n_slice_target)
            else:
                n_slice = n_slice_pred
                for sel_slice in range(n_slice):
                    print(f'Image number {i_img} / {n_img}. Slice number {sel_slice}  {n_slice}', end='\r')
                    x0 = self.us_array[i_img][sel_slice]
                    x = self.pred_array[i_img][sel_slice]
                    y = self.target_array[i_img][sel_slice]
                    # Expand on location..
                    # Perform difference minmax scaling..?
                    metric_dict = self._calculate(x, y, i_img=i_img)
                    metric_dict_input = self._calculate(x0, x, i_img=i_img, key_appendix='input')
                    metric_dict_target = self._calculate(x0, y, i_img=i_img, key_appendix='target')
                    metric_list.append(metric_dict)
                    metric_list.append(metric_dict_input)
                    metric_list.append(metric_dict_target)
        return metric_list

    def _calculate(self, x, y, i_img, key_appendix=''):
        x_minmax = harray.scale_minmax(x)
        y_minmax = harray.scale_minmax(y)
        x_meanstd = harray.scale_mean_std(x)
        y_meanstd = harray.scale_mean_std(y)
        ssim_value = structural_similarity(x_minmax, y_minmax, data_range=1)
        l2_norm = np.linalg.norm((x_meanstd - y_meanstd).ravel(), ord=2)
        mse_meanstd = np.mean((x_meanstd - y_meanstd).ravel() ** 2)
        mse_minmax = np.mean((x_minmax - y_minmax).ravel() ** 2)
        contrast_ssim_value = hmetric.get_contrast_ssim(x_minmax, y_minmax)
        temp = homog_measure.get_glcm_patch_object(x_meanstd, patch_size=x.shape[0] // 2, glcm_dist=list(range(1, 10, 2)))
        contrast_glcm_value = homog_measure.get_glcm_features(temp, feature_keys='contrast')['contrast']
        psnr_value = psnr(image_true=y_minmax, image_test=x_minmax, data_range=y_minmax.max())
        flip_value = compute_ldrflip(reference=np.array([y_minmax, y_minmax, y_minmax]),
                                     test=np.array([x_minmax, x_minmax, x_minmax])).mean()
        circular_mask = hmisc.circular_mask(y.shape[-2:])
        hi_value = homog_metric.get_hi_value_integral(x_minmax, circular_mask)
        metric_dict = {'ssim': ssim_value, 'l2': l2_norm, 'contrast_glcm': contrast_glcm_value, 'mse_meanstd': mse_meanstd,
                       'mse_minmax': mse_minmax, 'psnr': psnr_value, 'contrast_ssim': contrast_ssim_value,
                       'flip': flip_value, 'hi': hi_value, 'filename': self.file_list_pred[i_img]}
        if key_appendix:
            metric_dict = {k + f"_{key_appendix}": v for k, v in metric_dict.items()}

        return metric_dict

    def _plot(self):
        n_img = len(self.pred_array)
        for i_img in range(n_img):
            sel_slice_pred = self.pred_array[i_img].shape[0] // 2
            sel_slice_target = self.target_array[i_img].shape[0] // 2
            x = harray.scale_minmax(self.pred_array[i_img][sel_slice_pred])
            y = harray.scale_minmax(self.target_array[i_img][sel_slice_target])
            plot_obj = hplotc.ListPlot([x, y, x-y], cbar=True)
            plot_obj.figure.savefig(os.path.join(self.ddest, f'pred_target_pair_{i_img}.png'))
            plt.close('all')


class VisualizeFolder(FileGatherer):
    # pred_folder - should be something like... unet_CIRCUS_SCRATCH/25p/train_mixed/mixed/5x
    # Or in case of inference something like... varnet_CIRCUS/75p/train_mixed/undersampled
    # It should contain the .h5 (=ext) files
    def __init__(self, pred_folder, ext='h5', ddest=None, file_str_appendix='', inference_bool=False, debug=False,
                 calc_us=True, proper_scaling=False, retro_bool=False):
        super().__init__(pred_folder=pred_folder, ext=ext, inference_bool=inference_bool,
                         retro_bool=retro_bool, debug=debug, calc_us=calc_us)
        if ddest is None:
            self.ddest = self.data_pred
        else:
            self.ddest = ddest

        self.file_str_appendix = file_str_appendix
        self.proper_scaling = proper_scaling

    def plot(self):
        """
              Plot the undersampled input and predicted input...
              """
        # n_input = len(self.us_array)
        n_pred = len(self.pred_array)
        # Plot only the middle slice..
        plot_obj = hplotc.PlotCollage([x[x.shape[0]//2] for x in self.pred_array], self.ddest, n_display=n_pred, plot_type='array',
                                      text_box=False, only_str=True, height_offset=0, proper_scaling=self.proper_scaling)
        plot_obj.plot_collage(str_appendix=self.file_str_appendix)


def plot_on_axis(x_value, y_dict, ax, title, colormap='plasma', xlabel='Iterations'):
    plt_cm = plt.get_cmap(colormap)
    num_lines = 3
    color_list = [plt_cm(1. * i / (num_lines + 1)) for i in range(1, num_lines + 1)]
    ax.set_prop_cycle('color', color_list)

    for k, v in y_dict.items():
        # Remove unnecessary parts of the label..
        k = re.sub('calgary_campinas_', '', k)
        k = re.sub('_metric', '', k)
        if 'psnr' in k:
            twinx_ax = ax.twinx()
            twinx_ax.plot(x_value[:len(v)], v, label=k.upper())
            twinx_ax.set_ylabel('PSNR')
        else:
            ax.plot(x_value[:len(v)], v, label=k.upper())

    # ask matplotlib for the plotted objects and their labels
    # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend/10129461#10129461
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twinx_ax.get_legend_handles_labels()
    twinx_ax.legend(lines + lines2, labels + labels2, loc='upper left')

    ax.set_ylim(0, 1)
    twinx_ax.set_ylim(0, 50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('SSIM/VIF')
    ax.set_title(title)


def get_mapping_fbpconvnet(num_batch=None):
    """
    This one is not really used anymore...
    :param num_batch:
    :return:
    """
    from objective_configuration.reconstruction import DMAPPING_FBPCONVNET, DWEIGHTS_FBPCONVNET
    model_obj = FBPCONVNet()
    random_state_dict = model_obj.state_dict()

    mapping_pd = pd.read_csv(DMAPPING_FBPCONVNET)
    mat_obj = scipy.io.loadmat(DWEIGHTS_FBPCONVNET)
    layer_content = mat_obj['net']['layers'][0][0]
    recovered_state_dict = {}
    for i, irow in mapping_pd.iterrows():
        if 'running_var' in irow['torch names']:
            continue
        elif 'num_batches_tracked' in irow['torch names']:
            torch_key = irow['torch names'].strip()
            # Copy this value from the random state dict..
            if num_batch is None:
                recovered_state_dict[torch_key] = random_state_dict[torch_key]
            else:
                recovered_state_dict[torch_key] = torch.Tensor([num_batch])
        else:
            torch_key = irow['torch names'].strip()
            torch_size = tuple([int(x) for x in irow['torch size'].split(',')])
            mat_index_layer = int(irow['file index'])
            mat_index_array = int(irow['array index'])
            selected_layer = layer_content[0, mat_index_layer]
            weights_layer = selected_layer['weights'][0][0]
            select_array = weights_layer[0][mat_index_array]
            # When we are dealing with the running mean..
            # We are also filling in running_var when dealing with running_mean
            # The row of running_var is not considered, since this contains a 'None' value
            if 'running_mean' in torch_key:
                torch_key_var = re.sub('running_mean', 'running_var', torch_key)
                recovered_state_dict[torch_key] = torch.from_numpy(select_array[:, 0].reshape(torch_size))
                recovered_state_dict[torch_key_var] = torch.from_numpy(select_array[:, 1].reshape(torch_size))
                # print('mean mean', torch.mean(recovered_state_dict[torch_key]))
                # print('mean var', torch.mean(recovered_state_dict[torch_key_var]))
            else:
                mat_value = select_array.reshape(torch_size)
                recovered_state_dict[torch_key] = torch.from_numpy(mat_value)

    return recovered_state_dict


def get_sensitivity_map_radial_sampled_cardiac_data(path, n_size, min_x=175, min_y=110):
    """
    n_size is used as the crop size that is added to minx and miny
    :param n_size:
    :return:
    """
    # Preproc sense map
    cpx_sense_obj = read_cpx.ReadCpx(path)
    cpx_sense_obj.get_par_file()
    #  'FOV (ap,fh,rl) [mm]': '600.00 700.00 700.00',
    fov_ap = 600
    fov_fh = 700
    temp_sense = np.squeeze(cpx_sense_obj.get_cpx_img())
    n_coil, _, nz, ny, nx = temp_sense.shape
    sel_slice = 57
    temp_sense = temp_sense[:, :, :, :, sel_slice]
    factor_ap = fov_ap / nz
    factor_fh = fov_fh / ny

    temp_sense = harray.rescale_complex_array(temp_sense, (1, 1, factor_ap, factor_fh))
    sense_img0 = temp_sense[:, 0]
    sense_img1 = temp_sense[:, 1]
    sos_img1 = np.sqrt((np.abs(sense_img1) ** 2).sum(axis=0))

    # Find crop coords that allign the sense refscan and original scan...
    # min_x = 175
    # min_y = 110
    crop_coords = (min_x, min_x + n_size, min_y, min_y + n_size)
    cropped_sos_img1 = harray.apply_crop(sos_img1, crop_coords=crop_coords)
    # Apply these crops to the individual coil images
    cropped_sense_img0 = np.array([harray.apply_crop(x, crop_coords=crop_coords) for x in sense_img0])
    sense_img = cropped_sense_img0 / cropped_sos_img1
    return sense_img


def get_radial_sampled_cardiac_data(path, cardiac_phase=-1):
    """

    :return:
    """
    # Preproc radial map
    cpx_radial_obj = read_cpx.ReadCpx(path)
    # Information from the par file
    # cpx_radial_obj.get_par_file()
    #  'FOV (ap,fh,rl) [mm]': '350.00 8.00  350.00',
    fov_ap = 350
    fov_rl = 350
    temp_cardiac = np.squeeze(cpx_radial_obj.get_cpx_img())
    temp_cardiac = temp_cardiac[:, :, ::-1, ::-1]
    n_coil, ncard, ny, nx = temp_cardiac.shape
    # Scale image to 1 mm isotropic..
    ap_factor = fov_ap / ny
    rl_factor = fov_rl / nx
    temp_cardiac = harray.rescale_complex_array(temp_cardiac, (1, 1, ap_factor, rl_factor))
    sel_cpx_array = temp_cardiac[:, cardiac_phase]
    return sel_cpx_array


def get_cartesian_sampled_cardiac_data(path):
    """
    :return: Returns cardiac image array and kspace version
    """
    # Load it
    cartesian_array = hmisc.load_array(path, data_key='vartosave')
    # Select a specific cardiac phase
    sel_card = -1
    sel_cpx_array = np.squeeze(cartesian_array)[:, :, sel_card]
    kspace_array = harray.transform_image_to_kspace_fftn(sel_cpx_array)
    return sel_cpx_array, kspace_array


def visualize_traj_BART(x_traj):
    # Shape should (3, ny, nx)
    if x_traj.shape[0] == 2:
        # If we only have xy coordinates, add zero
        empty_dim = np.zeros(x_traj.shape[1:])[None]
        x_traj = np.concatenate([x_traj, empty_dim], axis=0)
    fig = plt.figure()
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d.scatter(x_traj[0].ravel().real, x_traj[1].ravel().real, x_traj[2].ravel().real)
    return fig


def nufft_rss(traj, kspace_data):
    import bart
    temp_grid = bart.bart(1, 'nufft -i -t', traj, kspace_data)
    recon_img = bart.bart(1, 'rss 8', temp_grid)
    return recon_img


def nufft_pics(traj, kspace_data, sense_data):
    import bart
    return bart.bart(1, 'pics -S -r0.001 -t', traj, kspace_data, sense_data)


def undersample_trajectory(img_size, n_points, p_undersample=20, max_factor=0.5):
    # Im still not sure if it is np.pi/2 or np.pi...
    # It should be 0.5, so this is good.
    max_spokes = int(max(img_size) * np.pi * max_factor)
    n_undersample = int((p_undersample / 100) * max_spokes)
    # Define trajectory for fully sampling
    trajectory_radial = sigpy.mri.radial(coord_shape=(n_undersample, n_points, 2), img_shape=img_size, golden=False)
    trajectory_radial = trajectory_radial.reshape(-1, 2)
#
    # Define undersampled trajectory, the same for ALL the coils
    # undersampled_trajectory = np.array(np.split(trajectory_radial, max_spokes))
    # We selecteren hier indices van de lijnen die we WEG willen hebben
    # Dus bij undersampled trajectory worden er n - n_undersampled lijnen op 'null' gezet
    # Zo behouden n_undersampled lijnen hun data
    # random_lines = np.random.choice(range(max_spokes), size=(max_spokes - n_undersample), replace=False)
    # undersampled_trajectory[random_lines] = None
    # undersampled_trajectory = undersampled_trajectory.reshape(-1, 2)
    # return undersampled_trajectory
    return trajectory_radial


def CIRCUS_to_spokes(selected_mask, debug=False):
    # # Convert a CIRCUS Mask to radial lines / angles
    nx, ny = selected_mask.shape
    mid_x = nx // 2
    mid_y = ny // 2
    coord_top = np.array([(0, x) for x in np.argwhere(selected_mask[0, :]).ravel()])
    coord_left = np.array([(x, 0) for x in np.argwhere(selected_mask[:, 0]).ravel()])
    coord_right = np.array([(nx - 1, x) for x in np.argwhere(selected_mask[-1, :]).ravel()])
    coord_bottom = np.array([(x, ny - 1) for x in np.argwhere(selected_mask[:, -1]).ravel()])

    # Validate the found coords
    if debug:
        fig_obj = hplotc.ListPlot([selected_mask])
        fig_obj.ax_list[0].scatter(coord_top[:, 1], coord_top[:, 0], color='r')
        fig_obj.ax_list[0].scatter(coord_left[:, 1], coord_left[:, 0], color='b')
        fig_obj.ax_list[0].scatter(coord_right[:, 1], coord_right[:, 0], color='g')
        fig_obj.ax_list[0].scatter(coord_bottom[:, 1], coord_bottom[:, 0], color='y')

    # Now calculate the angles...
    angle_list = []
    for i_coord in coord_top.tolist() + coord_left.tolist() + coord_right.tolist() + coord_bottom.tolist():
        relative_coord = np.array(i_coord) - np.array([mid_x, mid_y])
        x_coord, y_coord = relative_coord
        # Convert to radian
        degree = np.arctan2(y_coord, x_coord) * 180 / np.pi
        angle_list.append(degree)

    # Draw lines from the center outwards
    # Since the 'lines' in CIRCUS are not straight.
    y_line = np.linspace(0, ny // 2, ny)
    x_line = np.zeros(nx)
    single_line = np.vstack([x_line, y_line]).T
    rot_x = []
    print('Number of spokes ', len(angle_list))
    for i_degree in angle_list:
        theta = np.radians(i_degree)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, -s), (s, c)))
        temp = np.dot(single_line, rot_mat)
        rot_x.append(temp)
    res = np.vstack(rot_x)[:, ::-1]

    return res


def undersample_img(card_array,  traj, width=6, ovs=1.25):
    # width = 6
    # ovs = 1.25  # Why set it to 1.25...?
    # ovs = 1  # I am setting it to 1 since I think that will do nothing.
    dcf = np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2)
    input_array = []
    img_shape = card_array.shape[-2:]
    if card_array.ndim == 2:
        card_array = card_array[None]

    for i_coil in card_array:
        temp_kspace = sigpy.nufft(i_coil, coord=traj, width=width, oversamp=ovs)
        temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=traj, oshape=img_shape,
                                       width=width, oversamp=ovs)
        input_array.append(temp_img)

    input_array = np.array(input_array)
    return input_array


def undersample_imgv2(card_array,  traj, width=6, ovs=1.25):
    # Card array should be (..., ndim)
    # width = 6
    # ovs = 1.25  # Why set it to 1.25...?
    # ovs = 1  # I am setting it to 1 since I think that will do nothing.
    dcf = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
    img_shape = card_array.shape[-2:]
    if card_array.ndim == 2:
        card_array = card_array[None]

    temp_kspace = sigpy.nufft(card_array, coord=traj, width=width, oversamp=ovs)
    temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=traj, oshape=img_shape,
                                   width=width, oversamp=ovs)
    return temp_img


def undersample_to_kspace(card_array,  traj, width=4, ovs=1.25):
    # width = 6
    # ovs = 1.25  # Why set it to 1.25...? This is pretty standard it seems...
    dcf = np.sqrt(traj[:, 0] ** 2 + traj[:, 1] ** 2)
    kspace_array = []
    if card_array.ndim == 2:
        card_array = card_array[None]

    for i_coil in card_array:
        temp_kspace = sigpy.nufft(i_coil, coord=traj, width=width, oversamp=ovs) * dcf
        # temp_img = sigpy.nufft_adjoint(temp_kspace * dcf, coord=traj, oshape=img_shape,
        #                                width=width, oversamp=ovs)
        kspace_array.append(temp_kspace)

    kspace_array = np.array(kspace_array)
    return kspace_array


def get_square_ordered_idxs(square_side_size: int, square_id: int):
    """Returns ordered (clockwise) indices of a sub-square of a square matrix.

    Parameters
    ----------
    square_side_size: int
        Square side size. Dim of array.
    square_id: int
        Number of sub-square. Can be 0, ..., square_side_size // 2.

    Returns
    -------
    ordered_idxs: List of tuples.
        Indices of each point that belongs to the square_id-th sub-square
        starting from top-left point clockwise.
    """
    assert square_id in range(square_side_size // 2)

    ordered_idxs = list()

    for col in range(square_id, square_side_size - square_id):
        ordered_idxs.append((square_id, col))

    for row in range(square_id + 1, square_side_size - (square_id + 1)):
        ordered_idxs.append((row, square_side_size - (square_id + 1)))

    for col in range(square_side_size - (square_id + 1), square_id, -1):
        ordered_idxs.append((square_side_size - (square_id + 1), col))

    for row in range(square_side_size - (square_id + 1), square_id, -1):
        ordered_idxs.append((row, square_id))

    return tuple(ordered_idxs)


def circus_radial_mask(shape, acceleration):
    """Implements CIRCUS radial undersampling."""
    max_dim = max(shape) - max(shape) % 2
    min_dim = min(shape) - min(shape) % 2
    num_nested_squares = max_dim // 2
    M = int(np.prod(shape) / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))

    mask = np.zeros((max_dim, max_dim), dtype=np.float32)

    rng = np.random.RandomState()
    t = rng.randint(low=0, high=1e4, size=1, dtype=int).item()

    for square_id in range(num_nested_squares):
        ordered_indices = get_square_ordered_idxs(
            square_side_size=max_dim,
            square_id=square_id,
        )
        # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
        J = 2 * (num_nested_squares - square_id)
        # K: total number of points along the perimeter of the square K=4·J-4;
        K = 4 * (J - 1)

        for m in range(M):
            indices_idx = int(np.floor(np.mod((m + t * M) / GOLDEN_RATIO, 1) * K))
            mask[ordered_indices[indices_idx]] = 1.0

    pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

    mask = np.pad(mask, pad, constant_values=0)
    # mask = T.center_crop(torch.from_numpy(mask.astype(bool)), shape)

    return mask


def undersample_img_CIRCUS(card_array, acceleration):
    """
    Use the CIRCUS trajectory to undersample an image
    :param card_array:
    :return:
    """
    CIRCUS_mask = circus_radial_mask(card_array.shape[-2:], acceleration=acceleration)
    input_array = []
    if card_array.ndim == 2:
        card_array = card_array[None]
    for i_card in card_array:
        A_kspace = np.fft.fft2(i_card)
        A_img = np.fft.ifft2(A_kspace * CIRCUS_mask)
        input_array.append(A_img)

    return np.array(input_array)


def convert_to_sos(img):
    # Simple helper...
    # Because I find this so ugly..
    sos_img = np.sqrt((np.abs(np.fft.ifftn(img[..., ::2] + 1j * img[..., 1::2], axes=(-3, -2))) ** 2).sum(axis=-1))
    return sos_img


def convert_to_sos_npy(img):
    sos_array = np.sqrt(np.sum(np.abs(img) ** 2, axis=0))
    return sos_array


def prepare_metric_dataset(djson):
    if not os.path.isfile(djson):
        return None
    metric_dict = hmisc.load_json(djson)

    df_temp = harray.nested_dict_to_df(metric_dict, column_name='test')
    df_temp.index = df_temp.index.set_names(['percentage', 'dataset', 'anatomy', 'acceleration', 'metric'])
    prepped_table = df_temp.reset_index().pivot(columns='metric', values='test',
                                                index=['percentage', 'dataset', 'anatomy', 'acceleration'])

    # Reset index to convert the MultiIndex to columns
    df_reset = prepped_table.reset_index()
    df_reset['percentage'] = df_reset['percentage'].apply(lambda x: int(x[:-1]))
    df_reset['acceleration'] = df_reset['acceleration'].apply(lambda x: int(x[:-1]))
    return df_reset


def setup_metric_figure(metric, x_axis=False, inference_bool=False, big=False):
    ncol = 2
    if inference_bool:
        ncol = 1
    if big:
        fig, ax = plt.subplots(1, ncol, figsize=(40, 40), dpi=600)
    else:
        fig, ax = plt.subplots(1, ncol, figsize=(14, 10), dpi=600)

    plt.subplots_adjust(wspace=0.5)
    if inference_bool:
        ax = [ax]
        ax[0].set_title('5x acceleration: inference', fontsize=FONTSIZE_TITLE)
    else:
        ax[0].set_title('5x acceleration', fontsize=FONTSIZE_TITLE)
        ax[1].set_title('10x acceleration', fontsize=FONTSIZE_TITLE)

    for i_ax in ax:
        if x_axis:
            i_ax.tick_params(axis='x', which='major', labelsize=FONTSIZE_XTICKS)
        else:
            # Turn off the x-axis ticks
            i_ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
        i_ax.tick_params(axis='y', which='major', labelsize=FONTSIZE_YTICKS)
        if metric == 'ssim':
            y_max = 1
            y_step = 0.1
            i_ax.set_ylim(0, y_max)
            i_ax.set_yticks(np.arange(0, y_max, y_step), minor=True)
            i_ax.grid(which='both', zorder=0)
        elif metric == 'psnr':
            y_max = 30
            y_step = 1
            i_ax.set_ylim(0, y_max)
            i_ax.set_yticks(np.arange(0, y_max, y_step), minor=True)
            i_ax.grid(which='both', zorder=0)
        elif metric == 'l2':
            y_max = 30
            y_step = 1
            i_ax.set_ylim(0, y_max)
            i_ax.set_yticks(np.arange(0, y_max, y_step), minor=True)
            i_ax.grid(which='both', zorder=0)
        elif metric == 'mse_minmax':
            y_max = 0.01
            y_step = 0.0005
            i_ax.set_ylim(0, y_max)
            i_ax.set_yticks(np.arange(0, y_max, y_step), minor=True)
            i_ax.grid(which='both', zorder=0)
        elif metric == 'mse_meanstd':
            y_max = 0.4
            y_step = 0.05
            i_ax.set_ylim(0, y_max)
            i_ax.set_yticks(np.arange(0, y_max, y_step), minor=True)
            i_ax.grid(which='both', zorder=0)
        else:
            i_ax.grid(which='both', zorder=0)
    return fig, ax


def convert_direct2cpx(x):
    # Convert a loaded direct array to its cpx version with coils in first axis
    x_cpx = x[..., ::2] + 1j * x[..., 1::2]
    x_cpx = np.moveaxis(x_cpx, -1, 0)
    return x_cpx


def convert_direct2cpx_img(x):
    # Convert a loaded direct array to its cpx version with coils in first axis
    x_cpx = x[..., ::2] + 1j * x[..., 1::2]
    x_cpx = np.moveaxis(x_cpx, -1, 0)
    x_cpx = np.fft.ifft2(np.fft.ifftshift(x_cpx, axes=(-2, -1)))
    return x_cpx



def resize_array(x, target_size=(256, 256)):
    target_y, target_x = target_size
    nloc, ny, nx, ncoil = x.shape
    low_y = ny // 2 - target_y // 2
    high_y = ny // 2 + target_y // 2
    low_x = nx // 2 - target_x // 2
    high_x = nx // 2 + target_x // 2
    x_cropped = x[:, low_y: high_y, low_x: high_x, :]
    return x_cropped


def mat2direct(mat_array):
    # Taking in mat-array since the key to get the data is very important
    # This makes the order (ny, nx, ncoil, nloc)
    mat_swapped = np.squeeze(mat_array)
    A_direct = scan2direct_array(mat_swapped)
    return A_direct


def freebreathing2direct_array(x):
    ny, nx, ncoil, n_loc = x.shape
    temp_kspace = np.fft.fftn(x, axes=(0, 1))
    # Reshape to (ny, nx, ncoil, n_loc, 2)
    temp_kspace = np.stack([temp_kspace.real, temp_kspace.imag], axis=-1)
    # Move to (n_loc, ny, nx, ncoil, 2)
    temp_kspace = np.moveaxis(temp_kspace, -2, 0)
    # Reshape to (n_loc, ny, nx, 2 * ncoil)
    temp_kspace = temp_kspace.reshape((n_loc, nx, ny, 2 * ncoil))
    return temp_kspace


def scan2direct_array(x):
    ny, nx, ncoil, n_loc = x.shape
    temp_kspace = np.fft.fftn(x, axes=(0, 1))
    # Going to re-create the old data setting again
    temp_kspace = np.fft.fftshift(temp_kspace, axes=(0, 1))
    # Reshape to (ny, nx, ncoil, n_loc, 2)
    temp_kspace = np.stack([temp_kspace.real, temp_kspace.imag], axis=-1)
    # Move to (n_loc, ny, nx, ncoil, 2)
    temp_kspace = np.moveaxis(temp_kspace, -2, 0)
    # Reshape to (n_loc, ny, nx, 2 * ncoil)
    temp_kspace = temp_kspace.reshape((n_loc, nx, ny, 2 * ncoil))
    return temp_kspace

# Single coil
def step_by_step_plot(sel_coil, operator='np.abs', max_scale=0.8):
    dim = (-2, -1)
    step_1 = np.fft.ifftshift(sel_coil, axes=dim)
    step_2 = np.fft.ifftn(step_1, axes=dim)
    step_3 = np.fft.fftshift(step_2, axes=dim)
    step_0 = np.fft.ifftn(sel_coil, axes=dim)
    plot_array = [sel_coil, step_1, step_2, step_3, step_0, step_0 - step_2]
    vmin_array = [(0, max_scale * np.abs(x).max()) for x in plot_array]
    fig_obj = hplotc.ListPlot(plot_array,
                              augm=operator, col_row=(4, 4), cbar=True,
                              vmin=vmin_array)
    # Hoe kan het dat zowel step_0 gelijk is aan step_2... voor Calgary data..??
    # Dat betekend dat de ifftshift niks zou moeten doen.
    # Maar step_1 laat zien dat dit WEL degelijk wat doet...
    #
    return fig_obj


# Configuration creation files
def DIRECT_update_inference_dict(temp_yaml):
    # Deep copy is necessary
    single_val_dataset = copy.deepcopy(temp_yaml['validation']['datasets'][0])
    # Make sure we use CalagaryCampinas
    hmisc.set_nested(single_val_dataset, ['name'], 'CalgaryCampinas')
    # This makes use of the np.ones() mask
    hmisc.set_nested(single_val_dataset, ['transforms', 'masking', 'name'], 'CalgaryCampinas')
    hmisc.set_nested(single_val_dataset, ['transforms', 'random_augmentations', 'random_rotation'], False)
    hmisc.set_nested(single_val_dataset, ['transforms', 'random_augmentations', 'random_flip'], False)
    # hmisc.set_nested(single_val_dataset, ['transforms', 'estimate_sensitivity_maps'], False)
    # hmisc.set_nested(single_val_dataset, ['sensitivity_maps'], '/home/sharreve/local_scratch/mri_data/cardiac_radial_inference/sensitivity')
    # And set a description
    hmisc.set_nested(single_val_dataset, ['text_description'], 'SebInference')

    # Append, so that the validation index can be set to 2
    temp_yaml['validation']['datasets'].append(single_val_dataset)
    return temp_yaml


def DIRECT_update_config_dict(yaml_dict, ddata, anatomical_region, percentage, batch_size=4, number_of_iterations=1000):
    """

    :param yaml_dict:
    :param anatomical_region: 2ch ,sa, transverse, 4ch
    :param percentage:
    :return:
    """
    # Avoid any typo's
    assert str(anatomical_region) in ANATOMY_LIST or anatomical_region == 'mixed'
    assert int(percentage) in PERCENTAGE_LIST

    # This can set the training list
    training_dir = os.path.join(ddata, f'{anatomical_region}/train/train_{percentage}.lst')
    # Number of iterations is a variable
    hmisc.set_nested(yaml_dict, ['training', 'num_iterations'], number_of_iterations)
    hmisc.set_nested(yaml_dict, ['training', 'loss', 'losses'], [{'function': 'perc_loss', 'multiplier': 1.0},
                                                                 {'function': 'l1_loss', 'multiplier': 2.0},
                                                                 {'function': 'ssim_loss', 'multiplier': 1.0}])
    hmisc.set_nested(yaml_dict, ['training', 'lr'], 1e-4)
    hmisc.set_nested(yaml_dict, ['training', 'lr_gamma'], 0.2)
    hmisc.set_nested(yaml_dict, ['training', 'lr_step_size'], number_of_iterations)
    hmisc.set_nested(yaml_dict, ['training', 'lr_warmup_iter'], int(number_of_iterations/2))
    hmisc.set_nested(yaml_dict, ['training', 'validation_steps'], int(number_of_iterations / 10))

    # For all training and validation datasets, alter the following:
    for i_type in ['training', 'validation']:
        # Reduce this, because otherwise the GPU usages get too large...
        hmisc.set_nested(yaml_dict, [i_type, 'batch_size'], batch_size)
        for ii, i_dataset in enumerate(hmisc.get_nested(yaml_dict, [i_type, 'datasets'])):
            # This is needed to make sure that the further pipelining works..
            # Because index 0 and index 1 are linked to acc 5x and 10x.
            # This can be a possible source of error...
            if ii == 0:
                acceleration = 5
            else:
                acceleration = 10
            i_dataset['crop_outer_slices'] = False
            # This is not  CalagaryCampinas.
            # We have modified it such that everything is radially sampled (interpolated)
            hmisc.set_nested(i_dataset, ['name'], 'CalgaryCampinasSeb')
            # We keep CalgaryCampinas as masking because it does nothing
            hmisc.set_nested(i_dataset, ['transforms', 'masking', 'name'], 'CalgaryCampinas')
            hmisc.set_nested(i_dataset, ['transforms', 'masking', 'accelerations'], [acceleration])

            # This is needed to detect the acceleration factor
            hmisc.set_nested(i_dataset, ['text_description'], f'{str(acceleration)}x')
            if i_type == 'training':
                # Adding a random augmnetation section to training. Not validation because of misalignment with target
                hmisc.set_nested(i_dataset, ['transforms', 'random_augmentations', 'random_rotation'], True)
                hmisc.set_nested(i_dataset, ['transforms', 'random_augmentations', 'random_flip'], True)
                #
                i_dataset['filenames_lists'] = [training_dir]
                # Remove lists... We need filenames lists
                if 'lists' in i_dataset.keys():
                    del i_dataset['lists']
    return yaml_dict
