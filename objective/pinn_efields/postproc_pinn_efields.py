from objective.postproc_base import PostProcBase
import data_generator.PinnEfields as data_gen
import h5py
import os
import numpy as np
import helper.plot_class as hplotc
import torch


class PostProcPinnEfields(PostProcBase):
    def __init__(self, executor_module, ddest, config_dict=None, config_path=None, config_name='config_run.json', **kwargs):
        super().__init__(executor_module=executor_module, config_dict=config_dict, config_path=config_path, config_name=config_name, **kwargs)
        self.debug = kwargs.get('debug', False)
        if self.debug:
            print("Setting model object...")
        self.model_obj = self.set_model_object()
        if self.debug:
            print("Setting data loader...")
        self.test_loader = self.set_data_loader()
        self.ddest = ddest

    def set_model_object(self):
        model_choice = self.config_dict['model']['model_choice']
        model_obj = None
        if model_choice == 'regular':
            model_obj = self.modelrun_obj.model_obj
        elif model_choice == 'gan':
            model_obj = self.modelrun_obj.generator
        elif model_choice == 'cyclegan':
            model_obj = self.modelrun_obj.netG_A2B
        return model_obj

    def set_data_loader(self):
        # I assume this works.....
        num_workers = self.config_dict.get('data').get('num_workers', 16)
        test_loader = self.modelrun_obj.get_data_loader('test', num_workers=num_workers)
        return test_loader

    def store_single_result(self, index):
        # Because sometimes the visualization does not go as planned
        X, Y, Y_approx = self.evaluate_container(index)
        input_file = os.path.join(self.ddest, 'input.h5')
        target_file = os.path.join(self.ddest, 'target.h5')
        pred_file = os.path.join(self.ddest, 'pred.h5')
        with h5py.File(input_file, 'w') as f:
            f.create_dataset('data', data=X)
        with h5py.File(target_file, 'w') as f:
            f.create_dataset('data', data=Y)
        with h5py.File(pred_file, 'w') as f:
            f.create_dataset('data', data=Y_approx)

    def visualize_test_example(self, index=0):
        input_array, target_array, mask_array = self.get_X_Y_mask(index)
        input_array = input_array.numpy()
        target_array = target_array.numpy()
        mask_array = mask_array.numpy()
        input_array[np.isnan(input_array)] = 0
        fig_obj = hplotc.ListPlot([input_array, target_array, mask_array])
        fig_obj.figure.savefig(os.path.join(self.ddest, 'example_data.png'))

    def get_X_Y_mask(self, index):
        container = self.test_loader.dataset.__getitem__(index)
        X = container['input']
        Y = container['target']
        X_mask = container['mask']
        if X.ndim == 3:
            X = X[None]
            Y = Y[None]
            X_mask = X_mask[None]
        return X, Y, X_mask

    def evaluate_container(self, index):
        X, Y, X_mask = self.get_X_Y_mask(index)
        X = X.to(self.modelrun_obj.device)
        Y = Y.to(self.modelrun_obj.device)
        X_mask = X_mask.numpy()
        with torch.no_grad():
            result = self.model_obj(X)
        Y_approx = result.cpu().numpy()
        return X.cpu().numpy(), Y.cpu().numpy() * X_mask, Y_approx * X_mask

    def evaluate_test_loader(self, n_examples):
        n = len(self.test_loader)
        for index in range(n):
            X, Y, Y_approx = self.evaluate_container(index)
            fig_obj = hplotc.ListPlot([X, Y, Y_approx])
            fig_obj.figure.savefig(os.path.join(self.ddest, f'example_data_{index}.png'))
            if index > n_examples:
                break

