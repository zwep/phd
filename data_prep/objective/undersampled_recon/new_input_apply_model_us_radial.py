
import data_generator.UndersampledRecon as dg_us
import objective.undersampled_recon.executor_undersampled_recon as executor_radial
import helper.misc as hmisc
import os

import torch
import numpy as np
import skimage.transform as sktransf
import helper.array_transf as harray
import helper.plot_class as hplotc

"""
Because the undersampled radial stuff is not really alligned with the cardiac cartesian stuff..
We want to create a new dataset based on that...

So per dataset (4ch, transverse, 2ch) create a NEW data set with the input (undersampled radial) being
pushed through a model and aligned in mask-size with the cartesian data set.
Hopefully this all goes smoothly....

Same operations per data set
Same oeprations per test/train/validation
Same operations per input/output pair

Try to avoid rescaling..?

Post processings teps
Input -> to model..
Normalize

"""


class ProcessUndersampledRadial:
    def __init__(self, source_path, data_type, model_obj, debug=False):
        print('Dealing with ', source_path)
        print('..and... ', data_type)
        self.debug = debug

        self.model_obj = model_obj
        self.source_path = source_path
        self.data_type = data_type

        # for sel_type in dataset_types:
        self.input_dir = os.path.join(self.source_path, self.data_type, 'input')
        self.target_dir = os.path.join(self.source_path, self.data_type, 'target')
        self.file_list = os.listdir(self.input_dir)

        self.dest_dir = source_path + "_processed"
        self.proc_input_dir = os.path.join(self.dest_dir, self.data_type, 'input')
        self.proc_target_dir = os.path.join(self.dest_dir, self.data_type, 'target')

        if not os.path.isdir(self.proc_input_dir):
            os.makedirs(self.proc_input_dir)
        if not os.path.isdir(self.dest_dir):
            os.makedirs(self.dest_dir)
        if not os.path.isdir(self.proc_target_dir):
            os.makedirs(self.proc_target_dir)

        self.input_affine_coords = None
        self.input_crop_coords = None
        self.target_affine_coords = None
        self.target_crop_coords = None

    def load_files(self, input_file, target_file):
        input_array = np.load(input_file)
        target_array = np.load(target_file)

        n_coils, n_card_input, _, _ = input_array.shape
        n_card_target, _, _ = target_array.shape

        # Select only 8 coils for now.
        input_array = input_array[-8:]
        # Set to order (n_card, coils, x, y)
        input_array = np.moveaxis(input_array, 0, 1)

        target_array = target_array

        if self.debug:
            print('Loaded files')
            print('Shape input ', input_array.shape)
            print('Shape target ', target_array.shape)

        return input_array, target_array

    def get_input_tensor(self, temp_array):
        temp_array = harray.scale_minpercentile(temp_array, q=98, is_complex=True, axis=(-2, -1))
        temp_array = harray.scale_minmax(temp_array, is_complex=True, axis=(-2, -1))
        n_y, n_x = temp_array.shape[-2:]
        temp_stacked = harray.to_stacked(temp_array, cpx_type='cartesian', stack_ax=0)
        temp_stacked = temp_stacked.T.reshape((n_x, n_y, -1)).T
        temp_tensor = torch.as_tensor(temp_stacked[np.newaxis]).float()
        if self.debug:
            print('Input after being a tensor shape ', temp_tensor.shape)
        return temp_tensor

    def get_edges(self, profile):
        diff_profile = np.diff(profile).astype(int)
        loc_diff = np.argwhere(diff_profile)

        if self.debug:
            print('\n\n')
            print(diff_profile)
            print(loc_diff)
            print(loc_diff.shape, len(loc_diff), loc_diff.ndim)

        index0, index1 = (min(profile), max(profile))
        return index0, index1

    def set_scale_transformation(self, input_array, target_array):
        if self.debug:
            print('Set scale transformation')
            print('Shape input ', input_array.shape)
            print('Shape target ', target_array.shape)

        input_mask = harray.get_treshold_label_mask(input_array)
        input_mask, _ = harray.convex_hull_image(input_mask)
        target_mask = harray.get_treshold_label_mask(target_array)
        target_mask, _ = harray.convex_hull_image(target_mask)

        # Determine size difference based on mask....
        if 'transverse' in self.source_path:
            res_index0, res_index1 = self.get_edges(input_mask.sum(axis=1))
            cart_index0, cart_index1 = self.get_edges(target_mask.sum(axis=1))
        elif '2ch' in self.source_path:
            res_index0, res_index1 = self.get_edges(input_mask.sum(axis=0))
            cart_index0, cart_index1 = self.get_edges(target_mask.sum(axis=0))
        else:
            res_index0 = res_index1 = 0
            cart_index1 = cart_index0 = 0

        cart_width = cart_index1 - cart_index0
        res_width = res_index1 - res_index0
        # This is the old width, ratio and new width
        self.n_width = input_mask.shape[0]
        self.ratio = cart_width / res_width
        self.new_width = int(self.n_width * self.ratio)

    def set_affine_transformation(self, input_array, target_array):
        if self.debug:
            print('Set affine transformation')
            print('Shape input ', input_array.shape)
            print('Shape target ', target_array.shape)

        input_mask = harray.get_treshold_label_mask(input_array)
        input_mask, _ = harray.convex_hull_image(input_mask)
        target_mask = harray.get_treshold_label_mask(target_array)
        target_mask, _ = harray.convex_hull_image(target_mask)
        # We need the mask to determine the affine transfomration to put it in the middle..
        self.input_affine_coords, self.input_crop_coords = harray.get_center_transformation_coords(input_mask)
        self.target_affine_coords, self.target_crop_coords = harray.get_center_transformation_coords(target_mask)

    def resize_input(self, input_array, target_array):
        # Rescale it to the size of the target array
        # Either crop it.. or make a submatrix..

        temp_resize = sktransf.resize(input_array, (self.new_width, self.new_width), anti_aliasing=False)
        if self.ratio <= 1:
            cropped_input_array = np.zeros(target_array.shape)
            try:
                cropped_input_array[
                self.n_width // 2 - self.new_width // 2: self.n_width // 2 + self.new_width // 2 + 1,
                self.n_width // 2 - self.new_width // 2: self.n_width // 2 + self.new_width // 2 + 1] = temp_resize
            except:
                cropped_input_array[self.n_width // 2 - self.new_width // 2: self.n_width // 2 + self.new_width // 2,
                self.n_width // 2 - self.new_width // 2: self.n_width // 2 + self.new_width // 2] = temp_resize

        else:
            cropped_input_array = temp_resize[self.new_width // 2 - self.n_width // 2: self.new_width // 2 + self.n_width // 2,
                                self.new_width // 2 - self.n_width // 2: self.new_width // 2 + self.n_width // 2]

        return cropped_input_array

    def process_single_file(self, input_file_name):
        input_file = os.path.join(self.input_dir, input_file_name)
        target_file = os.path.join(self.target_dir, input_file_name)

        input_array_card, target_array_card = self.load_files(input_file=input_file, target_file=target_file)

        # Make it so that the cardiac phase is first...
        n_card_target = target_array_card.shape[0]
        n_card_input = input_array_card.shape[0]

        new_input_array = []
        new_target_array = []
        for ii, temp_input in enumerate(input_array_card):
            print('Starting cardiac phase ', ii)
            input_tensor = self.get_input_tensor(temp_input)

            sel_card_target = int(ii * n_card_target / n_card_input)
            target_array = np.abs(target_array_card[sel_card_target])
            target_array = harray.scale_minpercentile(target_array, q=98, axis=(-2, -1))
            target_array = harray.scale_minmax(target_array, axis=(-2, -1))

            if self.debug:
                print('Target array shape', target_array.shape)

            with torch.no_grad():
                temp_res = self.model_obj(input_tensor)

            processed_array = temp_res[0][0].numpy()
            processed_array = harray.scale_minmax(processed_array)

            # Only determine transformation parameters with the FIRST cardiac phase...
            # This is to make the transformations consisttent over all cardiahases
            if ii == 0:
                if ('transverse' in self.source_path) or ('2ch' in self.source_path):
                    self.set_scale_transformation(input_array=processed_array, target_array=target_array)
                    processed_array = self.resize_input(input_array=processed_array, target_array=target_array)

                # I think that setting the affine transform parameters AFTER the resizing is done is better...
                self.set_affine_transformation(input_array=processed_array, target_array=target_array)

            else:
                if ('transverse' in self.source_path) or ('2ch' in self.source_path):
                    processed_array = self.resize_input(input_array=processed_array, target_array=target_array)

            processed_array = harray.apply_center_transformation(processed_array, affine_coords=self.input_affine_coords,
                                                           crop_coords=self.input_crop_coords)

            new_target = harray.apply_center_transformation(target_array, affine_coords=self.target_affine_coords,
                                                            crop_coords=self.target_crop_coords)

            processed_array = harray.scale_minmax(processed_array)
            new_target = harray.scale_minmax(new_target)

            new_input_array.append(processed_array)
            new_target_array.append(new_target)

        new_input_array = np.array(new_input_array)
        new_target_array = np.array(new_target_array)
        return new_input_array, new_target_array

    def process_all_files(self):
        # Loop over the files.. not the iterator
        for index in range(len(self.file_list)):
            i_file = self.file_list[index]
            new_input_array, new_target_array = self.process_single_file(i_file)

            np.save(os.path.join(self.proc_input_dir, i_file), new_input_array)
            np.save(os.path.join(self.proc_target_dir, i_file), new_target_array)


if __name__ == "__main__":

    model_path = '/data/seb/model_run/undersampled_recon_radial_20/config_00'

    """
    Load model

    dont use the generator from this object... we rather create it ourselves
    """

    config_param = hmisc.convert_remote2local_dict(model_path, path_prefix='/media/bugger/MyBook/data/semireal')
    config_param['data']['batch_size'] = 1
    decision_obj = executor_radial.DecisionMakerRecon(config_file=config_param, debug=True, load_model_only=True,
                                                      inference=False, device='cpu')  # ==>>
    modelrun_obj = decision_obj.decision_maker()
    modelrun_obj.load_weights()
    if modelrun_obj.model_obj:
        modelrun_obj.model_obj.eval()
    else:
        modelrun_obj.generator.eval()

    """
    Load data

    Because we want to get ALL the cardiac phases... we are stuck with using copy-pasted code
    """

    ddata = '/data/seb/unfolded_radial/cartesian_radial_dataset_4ch;/data/seb/unfolded_radial/cartesian_radial_dataset_p2ch;/data/seb/unfolded_radial/cartesian_radial_dataset_transverse'
    dataset_source = ddata.split(";")
    dataset_types = ['test', 'validation', 'train']

    for i_source in dataset_source:
        print(i_source)
        for i_type in dataset_types:
            print(i_type)
            proc_obj = ProcessUndersampledRadial(source_path=i_source, data_type=i_type, model_obj=modelrun_obj.model_obj, debug=True)
            # new_input_array, new_target_array = proc_obj.process_single_file(proc_obj.file_list[0])
            # plot_obj = hplotc.ListPlot([new_input_array[::5], new_target_array[::5]])
            # plot_obj.figure.savefig(f'/data/seb/test_outcome/{os.path.basename(i_source)}_{proc_obj.data_type}')
            proc_obj.process_all_files()