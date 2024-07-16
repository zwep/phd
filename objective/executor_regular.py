# encoding: utf-8
import abc

import inspect
import json
import functools
import warnings

import numpy as np
import os

import torch
from matplotlib import pyplot as plt
import helper.plot_fun as hplotf
from helper_torch import misc as htmisc
import helper.misc as hmisc
import helper_torch.post_proc as htpostproc

from objective.executor_base import ExecutorBase
import data_generator.Generic as data_gen_generic
import GPUtil



class ExecutorRegular(ExecutorBase):
    def __init__(self, model_path=None, config_file=None, **kwargs):
        super().__init__(model_path=model_path, config_file=config_file, **kwargs)

        """
        Define the lambda functions for the losses
        """
        # Get the associate weighting-functions for this loss
        lambda_loss_config = self.config_param['model']['config_regular'].get('lambda_loss_config', {})
        self.lambda_loss = self.get_lambda_weight_fun(lambda_loss_config)

        # Get the associated weighting-function for this loss
        lambda_loss_additional_config = self.config_param['model']['config_regular'].get('lambda_loss_additional_config', {})
        self.lambda_loss_additional = self.get_lambda_weight_fun(lambda_loss_additional_config)

        lambda_xcnn_config = self.config_param['model']['config_regular'].get('lambda_xcnn_config', {})
        self.lambda_xcnn = self.get_lambda_weight_fun(lambda_xcnn_config)

        """
        Define the losses for the model
        """
        # Get the primary loss object
        loss_name = self.config_param['model']['config_regular']['loss']
        self.loss_config = self.config_param['model']['config_regular'].get('loss_config', {})
        self.loss_config.setdefault("run", {})
        self.loss_config.setdefault("obj", {})

        # Get the secondary loss object
        loss_additional_name = self.config_param['model']['config_regular'].get('loss_additional', '')
        self.loss_additional_config = self.config_param['model']['config_regular'].get('loss_additional_config', {})
        self.loss_additional_config.setdefault("run", {})
        self.loss_additional_config.setdefault("obj", {})

        if self.load_model_only is False:
            self.loss_obj = self.get_loss(loss_name=loss_name, options=self.loss_config['obj'])
            self.loss_additional_obj = self.get_loss(loss_name=loss_additional_name,
                                                     options=self.loss_additional_config['obj'])

        # Create data generators
        num_workers = self.config_param.get('data').get('num_workers', 16)

        if self.load_model_only is False:
            self.test_loader = self.get_data_loader('test', num_workers=num_workers)
            if self.inference is False:
                self.train_loader = self.get_data_loader('train', num_workers=num_workers)
                self.val_loader = self.get_data_loader('validation', num_workers=num_workers)

        # Get the model object
        model_choice = hmisc.get_nested(self.config_param, ['model', 'config_regular', 'network_choice'])
        config_model = hmisc.get_nested(self.config_param, ['model', f'config_{model_choice}'])
        self.model_obj = self.get_model(config_model=config_model, model_choice=model_choice)

        self.reload_weights_config = self.config_param['model']['config_regular'].get('reload_weights_config', {})
        if self.reload_weights_config.get('status', None):
            # Now an important assumptions comes...
            # That is that the weights of this path fit well with the current loaded model..
            # Well.. one way to find out
            path_to_weights = self.reload_weights_config['model_path']
            dir_temp_weight = os.path.join(path_to_weights, self.name_temp_weights)
            dir_model_weight = os.path.join(path_to_weights, self.name_model_weights)
            try:
                self.load_weights(dir_temp_weight=dir_temp_weight, dir_model_weight=dir_model_weight)
            except:
                print('Uh oh.. the weights dont fit...')

        # Get the optimizer
        self.optimizer_obj = self.get_optimizer(self.model_obj.parameters(), self.config_param['optimizer'])

        # Get the LR scheduler optimizer..
        self.scheduler_model_obj = self.get_lr_scheduler(self.optimizer_obj)

        # Get the pretrained model, if needed
        # For now leave it in... but make it a habit to preproccess the data that needs to
        # be treated by this model. Makes stuff easier and less time consuming.
        self.trained_model_config = self.config_param['data'].get('trained_model_config', {})
        if self.trained_model_config.get('status', None):
            model_path = self.trained_model_config['model_path']
            requires_grad = self.trained_model_config.get('requires_grad', False)
            print('Creating pretrained model')
            print(f'Currently on device {self.device}')
            print(f'Currently on device index {self.device.index}')

            self.trained_modelrun_obj = ExecutorBase(model_path=model_path, index_gpu=self.device.index, load_model_only=True)
            trained_config_param = self.trained_modelrun_obj.config_param
            trained_model_choice = hmisc.get_nested(trained_config_param, ['model', 'config_regular', 'network_choice'])
            config_model = hmisc.get_nested(trained_config_param, ['model', f'config_{trained_model_choice}'])
            self.trained_modelrun_obj.model_obj = self.trained_modelrun_obj.get_model(config_model=config_model, model_choice=trained_model_choice)
            print('Config param of pretrained model ')
            hmisc.print_dict(self.trained_modelrun_obj.config_param)
            path_to_weights = self.reload_weights_config['model_path']
            dir_temp_weight = os.path.join(path_to_weights, self.name_temp_weights)
            dir_model_weight = os.path.join(path_to_weights, self.name_model_weights)
            # Load weights...
            self.trained_modelrun_obj.model_obj = self.trained_modelrun_obj.load_weights_model(model_obj=self.trained_modelrun_obj.model_obj,
                                                                                               dir_temp_weight=dir_temp_weight,
                                                                                               dir_model_weight=dir_model_weight)
            # We dont want this to be trained again..
            # But we might in the future...?
            # Also for now it is a regular model object. But maybe a GAN/generator/A2B or something could also be an option
            # Simply depends on the type in the config.
            self.trained_modelrun_obj.set_requires_grad(network=self.trained_modelrun_obj.model_obj, requires_grad=requires_grad)

    def load_weights(self, dir_temp_weight=None, dir_model_weight=None):
        if dir_temp_weight is None:
            dir_temp_weight = os.path.join(self.config_param['dir']['doutput'], self.name_temp_weights)

        if dir_model_weight is None:
            dir_model_weight = os.path.join(self.config_param['dir']['doutput'], self.name_model_weights)

        # Loads the weights
        if self.model_obj is not None:
            if self.debug:
                print('We have found a model object')

            self.model_obj = self.load_weights_model(model_obj=self.model_obj,
                                                     dir_temp_weight=dir_temp_weight,
                                                     dir_model_weight=dir_model_weight,
                                                     device=self.device)
        else:
            print('ERROR - EXEC: \t Please, first create a model')

    def get_image_prediction(self, sel_batch=None, sel_item=None, **kwargs):
        # Predicts an image using the test set. and eval() mode.
        # Afterwards post-processes the images.
        if sel_batch is None:
            batch_range = min(1, len(self.test_loader))
            sel_batch = np.random.randint(batch_range)
        if sel_item is None:
            item_range = min(1, self.test_loader.batch_size)
            sel_item = np.random.randint(item_range)

        # Not doing anything with batch or item number...
        batch_counter = 0
        self.model_obj.eval()  # IMPORTANT
        print('Start predicting images for visualization..')
        with torch.no_grad():  # IMPORTANT
            for container in self.test_loader:
                if sel_batch == batch_counter:
                    print('Done predicting images for visualization')
                else:
                    print('Batch counter', sel_batch, '/', batch_counter, end='\r')
                    batch_counter += 1
                    continue

                if self.masked or \
                        self.loss_config['run'].get('masked', False) or \
                            self.loss_additional_config['run'].get('masked', False):
                    X = container['input']
                    y = container['target']
                    mask = container.get('mask', None)
                    if mask is None:
                        mask = container.get('mask_target', None)
                    # If it is still none....
                    if mask is None:
                        print('Sorry, cant find mask or mask_target')
                    else:
                        mask = mask.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X, y = X.to(self.device), y.to(self.device)

                if self.trained_model_config.get('status', False):
                    with torch.no_grad():
                        X = self.trained_modelrun_obj.model_obj(X)
                    # Scale it to 0..1 again... might be needed? Not fully tested, simply used it.
                    X = (X - torch.min(X)) / (torch.max(X) - torch.min(X))

                with torch.no_grad():
                    y_pred = self.model_obj(X)

                if self.masked:
                    if self.debug:
                        print('Prediction shape before mask multiplication: ', y_pred.shape)
                        print('Target shape before mask multiplication: ', y.shape)
                        print('Mask shape before mask multiplication: ', mask.shape)

                    y = y * mask
                    y_pred = y_pred * mask

                if sel_batch == batch_counter:
                    break

        self.model_obj.train()  # IMPORTANT
        output = self.postproc_output(X, y, y_pred, sel_item=sel_item)
        torch_input, torch_target, torch_pred, plot_augm = output
        return torch_input, torch_target, torch_pred, plot_augm

    def save_model(self, plot_name='final'):
        dir_weights = os.path.join(self.config_param['dir']['doutput'], self.name_model_weights)
        # dir_temp_weights = os.path.join(self.config_param['dir']['doutput'], self.name_temp_weights)
        dir_history = os.path.join(self.config_param['dir']['doutput'], self.name_model_hist)

        # Save weights - saving only model weights.. temp_weights should be done during training.
        torch.save(self.model_obj.state_dict(), dir_weights)
        # torch.save(self.temp_weights, dir_temp_weights)

        # Save history object
        serialized_json_history_obj = json.dumps(self.history_dict)
        with open(dir_history, 'w') as f:
            f.write(serialized_json_history_obj)

        x0, y0, y_pred0, plot_augmentation = self.get_image_prediction()
        print('Plot array shape of img pred ', x0.shape, y0.shape, y_pred0.shape)
        if x0.ndim > 2:
            # For now.. just assume that if we have a larger dimension.. then the amount of channels will always be in the first dim.
            n_chan = x0.shape[0]
        else:
            n_chan = -1
            print('Unkown output dimension of x0 ', x0.shape)

        doutput = self.config_param['dir']['doutput']

        # This should be a complex valued array...?
        # This is a really dirty trick to fix this issue...
        # Issue: stacking with shapes like (8, 256, 256) and (1, 256, 256) is not possible
        # But my function also accepts lists

        # Pick one batch...
        try:
            temp_division = y_pred0[0:1]/(x0+1e-4)
            temp_division[temp_division < -1] = -1
            temp_division[temp_division > 1] = 1
        except ValueError:
            print('Division was not possible.')
            temp_division = np.zeros((10, 10))

        try:
            plot_array = np.stack([x0, y0, y_pred0, temp_division], axis=0)
            print('Shape of plot array', plot_array.shape)
        except:
            plot_array = [x0, y0, y_pred0, temp_division]
            for iplot in plot_array:
                print('Shape of plot array', iplot.shape)

        # Subtitle was removed because of change in sizes etc...
        subtitle = [['input'] * n_chan, ['target'] * n_chan, ['pred'] * n_chan, ['division'] * n_chan]
        for i_augm in plot_augmentation:
            output_path = os.path.join(doutput, f"{plot_name}_{i_augm}.jpg")
            fig_handle = hplotf.plot_3d_list(plot_array, figsize=(15, 10), dpi=75, augm=i_augm,
                                             #subtitle=subtitle,
                                             title=i_augm)
            fig_handle.savefig(output_path)

        return fig_handle

    def save_history_graph(self):
        dir_history_graph = os.path.join(self.config_param['dir']['doutput'], self.name_loss_graph)

        n = max(len(self.history_dict['train_loss']) - 1, 1)
        test_loss = self.history_dict.get('test_loss', 0)

        fig_handle, ax = plt.subplots()
        ax.plot(self.history_dict['val_loss'], 'b-x', label='validation loss', alpha=0.5)
        ax.plot(self.history_dict['train_loss'], 'r-x', label='training loss', alpha=0.5)
        if 'loss_value' in self.history_dict.keys():
            ax.plot(self.history_dict['loss_value'], 'k--', label='loss value', alpha=0.4)
        if 'loss_value_additional' in self.history_dict.keys():
            ax.plot(self.history_dict['loss_value_additional'], 'k-.', label='loss value additional', alpha=0.4)

        ax.plot(n * [None] + [test_loss], 'k-x', label='test loss')
        ax.set_title('validation loss vs train loss vs test loss')
        plt.legend()

        fig_handle.savefig(dir_history_graph)
        return fig_handle

    def test_model(self, **kwargs):
        if self.model_obj is None:
            warnings.warn('EXEC - TEST: \t Please run self.get_model() first')

        self.model_obj.eval()  # IMPORTANT
        running_loss = []
        with torch.no_grad():  # IMPORTANT
            for container in self.test_loader:
                if self.masked or self.loss_config['run'].get('masked', False) or self.loss_additional_config['run'].get('masked', False):
                    X = container['input']
                    y = container['target']
                    mask = container.get('mask', None)
                    if mask is None:
                        mask = container.get('mask_target', None)

                    if mask is None:
                        print('Sorry, cant find mask or mask_target')
                    else:
                        mask = mask.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X, y = X.to(self.device), y.to(self.device)

                if self.debug:
                    print('Testing \t Put container content to device ')

                if self.trained_model_config.get('status', False):
                    X = self.trained_modelrun_obj.model_obj(X)

                y_pred = self.model_obj(X)

                if self.masked:
                    y_pred = y_pred * mask
                    y = y * mask


                # Ferry ugly...... only on loss for now.. maybe later for additional one as wel...
                if self.loss_config['run'].get('masked', False):
                    if self.loss_config['run'].get('target_is_difference', False):
                        loss = self.loss_obj(y_pred, y - X, mask=mask) * self.lambda_loss(999)
                    elif self.loss_config['run'].get('input_to_loss', False):
                        loss = self.loss_obj(y_pred, y, mask=mask, x_input=X) * self.lambda_loss(999)
                    else:
                        loss = self.loss_obj(y_pred, y, mask=mask) * self.lambda_loss(999)
                else:
                    if self.loss_config['run'].get('target_is_difference', False):
                        loss = self.loss_obj(y_pred, y - X) * self.lambda_loss(999)
                    elif self.loss_config['run'].get('input_to_loss', False):
                        loss = self.loss_obj(y_pred, y, x_input=X) * self.lambda_loss(999)
                    else:
                        loss = self.loss_obj(y_pred, y) * self.lambda_loss(999)

                if self.loss_additional_obj is not None:
                    if self.loss_additional_config['run'].get('masked', False):
                        # We call this on the 999th epoch, then we are sure that we have everything how we want it
                        temp_loss = self.loss_additional_obj(y_pred, y, mask=mask) * self.lambda_loss_additional(999)
                    elif self.loss_additional_config['run'].get('input_to_loss', False):
                        temp_loss = self.loss_additional_obj(y_pred, y, x_input=X) * self.lambda_loss_additional(999)
                    elif self.loss_additional_config['run'].get('target_is_difference', False):
                        temp_loss = self.loss_additional_obj(y_pred, y - X) * self.lambda_loss_additional(999)
                    else:
                        # We call this on the 999th epoch, then we are sure that we have everything how we want it
                        temp_loss = self.loss_additional_obj(y_pred, y) * self.lambda_loss_additional(999)


                    print(f'Loss component {loss.item()}, {temp_loss.item()}', end='\r')
                    loss += temp_loss

                running_loss.append(loss.item())

        self.test_loss = np.mean(running_loss)
        self.history_dict['test_loss'] = self.test_loss

        self.model_obj.train()

        return running_loss

    def train_model_mixed_precision(self, **kwargs):
        import torch.cuda.amp
        scaler = torch.cuda.amp.GradScaler()

        if self.debug:
            print('Training \t Using optimizer ', self.optimizer_obj)
            print('Training \t Using loss ', self.loss_obj)

        min_epoch_loss = 9999
        breakdown_counter = 0

        epoch_loss_curve = []
        loss_value_curve = []
        loss_additional_curve = []
        val_loss_curve = []

        self.model_obj.train()  # IMPORTANT
        n_epoch = self.config_param['model']['n_epoch']
        return_gradients = self.config_param['model'].get('return_gradients', False)
        n_epoch_save = self.config_param['model'].get('n_epoch_save', int(0.10 * n_epoch))

        ind_weight_clip = self.config_param['model']['config_regular'].get('indicator_clipweights', False)
        # Default value that is way to high of course...
        weight_clip = self.config_param['model']['config_regular'].get('clipweights_value', 99)

        if return_gradients:
            self.history_dict.setdefault('overal_grad', [])
            self.history_dict.setdefault('overal_param', [])

        debug_display_counter = 0
        epoch = 0
        while epoch < n_epoch and breakdown_counter < self.breakdown_limit:
            try:
                print(f"Training \t Epoch {epoch + 1}/{n_epoch} ...")
                print(f'Training \t Breakdown counter {breakdown_counter}')

                # Train
                epoch_loss = 0
                loss_value = 0
                loss_value_additional = 0
                if self.debug:
                    print('Training \t Start training process')

                if self.debug_cuda:
                    print('\nStart of training. Current utilization')
                    GPUtil.showUtilization()

                for container in self.train_loader:
                    # If we want to apply the transofmraiton resize... only then apply this `set_transofmraiton_param`
                    if self.config_param['data'].get('transform_resize', False):
                        self.train_loader.dataset.set_resize_parameters()

                    # Start with setting gradients to zero..
                    self.optimizer_obj.zero_grad(set_to_none=True)

                    if self.debug and debug_display_counter == 0:
                        print('Training \t First batch loaded')

                    if self.masked:
                        mask = container['mask']
                        mask = mask.to(self.device)

                    X = container['input']
                    y = container['target']

                    X, y = X.to(self.device), y.to(self.device)

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Put container content to device ')

                    if self.debug_cuda:
                        print('\nAfter loading data to GPU in training')
                        GPUtil.showUtilization()

                    if self.trained_model_config.get('status', False):
                        X = self.trained_modelrun_obj.model_obj(X)

                    # MIXED PRECISION
                    with torch.cuda.amp.autocast():
                        y_pred = self.model_obj(X)

                        if self.masked:
                            y_pred = y_pred * mask
                            y = y * mask

                        if self.debug and debug_display_counter == 0:
                            print('Training \t Batch prediction done')

                        if self.loss_config['run'].get('masked', False):
                            loss = self.loss_obj(y_pred, y, mask=mask) * self.lambda_loss(epoch)
                        else:
                            loss = self.loss_obj(y_pred, y) * self.lambda_loss(epoch)

                        if self.loss_additional_obj is not None:
                            if self.loss_additional_config['run'].get('masked', False):
                                temp_loss = self.loss_additional_obj(y_pred, y, mask=mask) * self.lambda_loss_additional(
                                    epoch)
                            elif self.loss_additional_config['run'].get('input_to_loss', False):
                                temp_loss = self.loss_additional_obj(y_pred, y, x_input=X) * self.lambda_loss_additional(epoch)
                            elif self.loss_additional_config['run'].get('target_is_difference', False):
                                temp_loss = self.loss_additional_obj(y_pred, y - X) * self.lambda_loss_additional(epoch)
                            else:
                                temp_loss = self.loss_additional_obj(y_pred, y) * self.lambda_loss_additional(epoch)

                            temp_loss_value = temp_loss.item()
                        else:
                            temp_loss = 0
                            temp_loss_value = 0

                        print(f'Training \t Loss: {float(loss.item()): .4f} Additional loss: {float(temp_loss_value): .4f}')

                        if self.debug_cuda:
                            print('\nPassing through one loss in training')
                            GPUtil.showUtilization()

                    # Store both losses seperately as well
                    loss_value += loss.item()
                    loss_value_additional += temp_loss_value
                    loss += temp_loss

                    # Caluclating additional loss over xCNN layers
                    list_children = list(self.model_obj.children())
                    for layer in list_children:
                        named_param = list(layer.named_parameters())
                        len_param = len(named_param)
                        if len_param:
                            for param_name, x_param in named_param:
                                if 'xcnn_weights' in param_name:
                                    corr = self.corrcoef(x_param.reshape(x_param.shape[0], -1))
                                    temp_loss = torch.sum(
                                        torch.abs(corr - torch.eye(corr.shape[0]).to(self.device))) * self.lambda_xcnn(epoch)
                                    if self.debug and debug_display_counter == 0:
                                        print('Exec regular - loss xcnn', temp_loss)
                                    loss += temp_loss

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Loss after XCNN update', loss)

                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Calculate gradients')

                    # MIXED PRECISION
                    scaler.scale(loss).backward()

                    if self.debug_cuda:
                        print('\nAfter backward() call in training')
                        GPUtil.showUtilization()

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Perform optimizer step')

                    # MIXED PRECISION
                    scaler.step(self.optimizer_obj)
                    scaler.update()

                    if self.debug_cuda:
                        print('\nPerform step with optimizer in training')
                        GPUtil.showUtilization()

                    epoch_loss += loss.item()

                    if ind_weight_clip:
                        for p in self.model_obj.parameters():
                            p.data.clamp_(-weight_clip, weight_clip)

                    if return_gradients:
                        list_children = htmisc.get_all_children(self.model_obj, [])
                        sel_layer_name, sel_layer_param = htmisc.get_all_parameters(list_children)

                        grad_level = htmisc.get_grad_layers(sel_layer_param, sel_layer_name, debug=self.debug,
                                                            debug_display_counter=debug_display_counter)
                        param_level = htmisc.get_param_layers(sel_layer_param, sel_layer_name, debug=self.debug)
                        grad_name, grad_array = zip(*grad_level)
                        param_name, param_array = zip(*param_level)
                        grad_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in grad_array]
                        param_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in param_array]
                        # print('grad per layer', len(grad_per_layer))

                        # print('grad per layer', len(grad_per_layer))
                        self.history_dict['overal_grad'].append(grad_per_layer)
                        self.history_dict['overal_param'].append(param_per_layer)
                        if self.debug_cuda:
                            print('\nAfter calculating the return gradients() in training')
                            GPUtil.showUtilization()

                    debug_display_counter += 1

                if self.debug_cuda:
                    print('\nBefore validation step')
                    GPUtil.showUtilization()

                val_loss = self.validate_model(epoch)
                val_loss_curve.append(val_loss)

                if self.debug_cuda:
                    print('\nAfter validation step')
                    GPUtil.showUtilization()

                # Average over amount of batches..
                epoch_loss = epoch_loss / self.train_loader.__len__()
                loss_value = loss_value / self.train_loader.__len__()
                loss_value_additional = loss_value_additional / self.train_loader.__len__()
                loss_value_curve.append(loss_value)
                loss_additional_curve.append(loss_value_additional)
                print(f'Training \t Average loss over batch: {float(epoch_loss): .4f}')

                if epoch_loss < min_epoch_loss:
                    if self.debug:
                        print(f'Training \t Old minimum loss {float(min_epoch_loss): .4f}')
                        print(f'Training \t New minimum loss {float(epoch_loss): .4f}')
                    min_epoch_loss = epoch_loss
                    self.temp_weights = self.model_obj.state_dict()

                # We want to have a decrease in loss...
                if epoch_loss != 0:
                    # For all the non-zero losses...
                    temp_curve = [x for x in val_loss_curve if x != 0]
                    historic_curve = temp_curve[-(self.memory_time + self.memory_length): -self.memory_time]
                    if len(historic_curve):
                        historic_loss = np.mean(historic_curve)
                    else:
                        # Some big number....
                        historic_loss = 9999

                    current_loss = np.mean(temp_curve[-self.memory_length:])
                    criterion = historic_loss - current_loss
                    print(f'validation loss criterion {criterion}\n'
                          f'historic loss {historic_loss} \n'
                          f'validation loss {val_loss}')

                    # If the current loss gets larger than the historic loss...
                    if criterion < 0:
                        breakdown_counter += 1
                        if self.debug:
                            print('Training \t Not improved, counter: ', breakdown_counter, '/', self.breakdown_limit)
                    else:
                        # Rrrrandom walk!
                        breakdown_counter -= 1
                        breakdown_counter = max(breakdown_counter, 0)
                        if self.debug:
                            print(f'Training \t Improved!')
                            print(f'         \t\t Historic mean loss over time period: {float(historic_loss): .4f}')
                            print(f'         \t\t Current mean loss over epochs: {float(current_loss): .4f}')

                epoch += 1

                # Every now and then.. save intermediate results..
                if epoch % max(n_epoch_save, 1) == 0:
                    dir_temp_weights = os.path.join(self.config_param['dir']['doutput'], self.name_temp_weights)
                    torch.save(self.temp_weights, dir_temp_weights)
                    plt.close('all')
                    self.save_model(plot_name='intermediate')

            except KeyboardInterrupt:
                print('\t\t Keyboard interrupt ')
                break

        if breakdown_counter > self.breakdown_limit:
            print('Training \t We are not advancing fast enough. Broke out of training loop')
        elif epoch >= n_epoch:
            print('Training \t Completed all epochs')
        else:
            print('Training \t Increase in loss.. break down')

        self.history_dict['train_loss'] = epoch_loss_curve  # batch_loss_curve
        self.history_dict['loss_value'] = loss_value_curve  # first loss curve.
        self.history_dict['loss_value_additional'] = loss_additional_curve  # second loss curve
        self.history_dict['val_loss'] = val_loss_curve

        return self.history_dict

    def train_model(self, **kwargs):
        """
        Here We have an example of using k-fold cross validation...

        We should simply re-define the Data Loaders in a loop
        # define a cross validation function

def crossvalid(model=None,criterion=None,optimizer=None,dataset=None,k_fold=5):

    train_score = pd.Series()
    val_score = pd.Series()

    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
#         print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)"
#               % (trll,trlr,trrl,trrr,vall,valr))

        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))

        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)

#         print(len(train_set),len(val_set))
#         print()

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                          shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=50,
                                          shuffle=True, num_workers=4)
        train_acc = train(res_model,criterion,optimizer,train_loader,epoch=1)
        train_score.at[i] = train_acc
        val_acc = valid(res_model,criterion,optimizer,val_loader)
        val_score.at[i] = val_acc

    return train_score,val_score



        :param kwargs:
        :return:
        """
        # :param kwargs: if you need to pass more arguments to the compiler
        if self.model_obj is None:
            warnings.warn('Please run self.get_model() first')

        if self.optimizer_obj is None:
            warnings.warn('Please define an optimizer_obj first')

        if self.debug:
            print('Training \t Using optimizer ', self.optimizer_obj)
            print('Training \t Using loss ', self.loss_obj)

        min_epoch_loss = 9999
        breakdown_counter = 0

        epoch_loss_curve = []
        loss_value_curve = []
        loss_additional_curve = []
        val_loss_curve = []

        self.model_obj.train()  # IMPORTANT
        n_epoch = self.config_param['model']['n_epoch']
        n_epoch_save = self.config_param['model'].get('n_epoch_save', int(0.10 * n_epoch))

        return_gradients = self.config_param['model'].get('return_gradients', False)
        ind_weight_clip = self.config_param['model'].get('indicator_clipweights', False)
        weight_clip = self.config_param['model'].get('clipweights_value', 99)

        if return_gradients:
            self.history_dict.setdefault('overal_grad', [])
            self.history_dict.setdefault('overal_param', [])

        pred_nan_found = False
        debug_display_counter = 0
        epoch = 0
        while epoch < n_epoch and breakdown_counter < self.breakdown_limit:
            try:
                print(f"Training \t Epoch {epoch + 1}/{n_epoch} ...")
                print(f'Training \t Breakdown counter {breakdown_counter}')

                # Train
                epoch_loss = 0
                loss_value = 0
                loss_value_additional = 0
                if self.debug:
                    print('Training \t Start training process')

                if self.debug_cuda:
                    print('\nStart of training. Current utilization')
                    GPUtil.showUtilization()

                for container in self.train_loader:
                    # If we want to apply the transofmraiton resize... only then apply this `set_transofmraiton_param`
                    if self.config_param['data'].get('transform_resize', False):
                        self.train_loader.dataset.set_resize_parameters()

                    # Start with setting gradients to zero..
                    self.optimizer_obj.zero_grad()

                    if self.debug and debug_display_counter == 0:
                        print('Training \t First batch loaded')

                    if self.masked or self.loss_config['run'].get('masked', False) or self.loss_additional_config['run'].get('masked', False):
                        mask = container.get('mask', None)
                        if mask is None:
                            mask = container.get('mask_target', None)

                        if mask is None:
                            print('Sorry, cant find mask or mask_target')
                        else:
                            mask = mask.to(self.device)

                    X = container['input']
                    y = container['target']

                    X, y = X.to(self.device), y.to(self.device)

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Put container content to device ')

                    if self.debug_cuda:
                        print('\nAfter loading data to GPU in training')
                        GPUtil.showUtilization()

                    if self.trained_model_config.get('status', False):
                        X = self.trained_modelrun_obj.model_obj(X)
                        tens_max = torch.amax(X, dim=(-3, -2, -1), keepdim=True)
                        tens_min = torch.amin(X, dim=(-3, -2, -1), keepdim=True)
                        X = (X - tens_min) / (tens_max - tens_min) 

                    y_pred = self.model_obj(X)

                    #if any(torch.isnan(y_pred.detach()).flatten()):
                     #   pred_nan_found = True
                        # Just do this...
                      #  breakdown_counter = 100 + self.breakdown_limit

                    if self.masked:
                        y_pred = y_pred * mask
                        y = y * mask

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Batch prediction done')

                    # Ferry ugly...... only on loss for now.. maybe later for additional one as wel...
                    if self.loss_config['run'].get('masked', False):
                        if self.loss_config['run'].get('target_is_difference', False):
                            loss = self.loss_obj(y_pred, y - X, mask=mask) * self.lambda_loss(epoch)
                        elif self.loss_config['run'].get('input_to_loss', False):
                            loss = self.loss_obj(y_pred, y, mask=mask, x_input=X) * self.lambda_loss(epoch)
                        else:
                            loss = self.loss_obj(y_pred, y, mask=mask) * self.lambda_loss(epoch)
                    else:
                        if self.loss_config['run'].get('target_is_difference', False):
                            loss = self.loss_obj(y_pred, y - X) * self.lambda_loss(epoch)
                        elif self.loss_config['run'].get('input_to_loss', False):
                            loss = self.loss_obj(y_pred, y, x_input=X) * self.lambda_loss(epoch)
                        else:
                            loss = self.loss_obj(y_pred, y) * self.lambda_loss(epoch)

                    if self.loss_additional_obj is not None:
                        if self.loss_additional_config['run'].get('masked', False):
                            temp_loss = self.loss_additional_obj(y_pred, y, mask=mask) * self.lambda_loss_additional(epoch)
                        elif self.loss_additional_config['run'].get('input_to_loss', False):
                            temp_loss = self.loss_additional_obj(y_pred, y, x_input=X) * self.lambda_loss_additional(epoch)
                        elif self.loss_additional_config['run'].get('target_is_difference', False):
                            temp_loss = self.loss_additional_obj(y_pred, y - X) * self.lambda_loss_additional(epoch)
                        else:
                            temp_loss = self.loss_additional_obj(y_pred, y) * self.lambda_loss_additional(epoch)

                        temp_loss_value = temp_loss.item()
                    else:
                        temp_loss = 0
                        temp_loss_value = 0

                    print(f'Training \t Loss: {float(loss): .4f} Additional loss: {float(temp_loss): .4f}', end='\r')

                    if self.debug_cuda:
                        print('\nPassing through one loss in training')
                        GPUtil.showUtilization()

                    # Store both losses seperately as well
                    loss_value += loss.item()
                    loss_value_additional += temp_loss_value
                    loss += temp_loss

                    # Caluclating additional loss over xCNN layers
                    # Maybe put this on or off..?
                    list_children = list(self.model_obj.children())
                    for layer in list_children:
                        named_param = list(layer.named_parameters())
                        len_param = len(named_param)
                        if len_param:
                            for param_name, x_param in named_param:
                                if 'xcnn_weights' in param_name:
                                    corr = self.corrcoef(x_param.reshape(x_param.shape[0], -1))
                                    temp_loss = torch.sum(torch.abs(corr - torch.eye(corr.shape[0]).to(self.device))) * self.lambda_xcnn(epoch)
                                    if self.debug and debug_display_counter == 0:
                                        print('Exec regular - loss xcnn', temp_loss)
                                    loss += temp_loss
                    
                    if self.debug and debug_display_counter == 0:
                        print(f'Training \t Loss after XCNN update {float(loss.item()): .4f}')

                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Calculate gradients')

                    loss.backward()
                    if self.debug_cuda:
                        print('\nAfter backward() call in training')
                        GPUtil.showUtilization()

                    # Added weight clamping..
                    # for p in self.model_obj.parameters():
                    #     p.data.clamp_(-0.01, 0.01)

                    if self.debug and debug_display_counter == 0:
                        print('Training \t Perform optimizer step')

                    self.optimizer_obj.step()
                    if self.debug_cuda:
                        print('\nPerform step with optimizer in training')
                        GPUtil.showUtilization()
                    # batch_loss_curve.append(loss.item())
                    epoch_loss += loss.item()

                    if ind_weight_clip:
                        for p in self.model_obj.parameters():
                            p.data.clamp_(-weight_clip, weight_clip)

                    if return_gradients:
                        list_children = htmisc.get_all_children(self.model_obj, [])
                        grad_per_layer, param_per_layer = self.get_param_gradients(list_children)

                        self.history_dict['overal_grad'].append(grad_per_layer)
                        self.history_dict['overal_param'].append(param_per_layer)

                        if self.debug_cuda:
                            print('\nAfter calculating the return gradients() in training')
                            GPUtil.showUtilization()

                    debug_display_counter += 1

                print()

                if self.debug_cuda:
                    print('\nBefore validation step')
                    GPUtil.showUtilization()

                val_loss = self.validate_model(epoch)
                val_loss_curve.append(val_loss)

                if self.debug_cuda:
                    print('\nAfter validation step')
                    GPUtil.showUtilization()

                # Average over amount of batches..
                epoch_loss = epoch_loss / self.train_loader.__len__()
                loss_value = loss_value / self.train_loader.__len__()
                loss_value_additional = loss_value_additional / self.train_loader.__len__()
                loss_value_curve.append(loss_value)
                loss_additional_curve.append(loss_value_additional)
                print(f'Training \t Average loss over batch: {float(epoch_loss): .4f}')

                if epoch_loss < min_epoch_loss:
                    if self.debug:
                        print(f'Training \t Old minimum loss {float(min_epoch_loss): .4f}')
                        print(f'Training \t New minimum loss {float(epoch_loss): .4f}')
                    min_epoch_loss = epoch_loss
                    self.temp_weights = self.model_obj.state_dict()

                # We want to have a decrease in loss...
                if epoch_loss != 0:
                    # For all the non-zero losses...
                    temp_curve = [x for x in val_loss_curve if x != 0]
                    historic_curve = temp_curve[-(self.memory_time + self.memory_length): -self.memory_time]
                    if len(historic_curve):
                        historic_loss = np.mean(historic_curve)
                    else:
                        # Some big number....
                        historic_loss = 9999

                    current_loss = np.mean(temp_curve[-self.memory_length:])
                    criterion = historic_loss - current_loss
                    print(f'validation loss criterion {criterion}\n'
                          f'historic loss {historic_loss} \n'
                          f'validation loss {val_loss}')

                    # If the current loss gets larger than the historic loss...
                    if criterion < 0:
                        breakdown_counter += 1
                        if self.debug:
                            print('Training \t Not improved, counter: ', breakdown_counter, '/', self.breakdown_limit)
                    else:
                        # Rrrrandom walk!
                        breakdown_counter -= 1
                        breakdown_counter = max(breakdown_counter, 0)
                        if self.debug:
                            print(f'Training \t Improved!')
                            print(f'         \t\t Historic mean loss over time period: {float(historic_loss): .4f}')
                            print(f'         \t\t Current mean loss over epochs: {float(current_loss): .4f}')

                epoch += 1

                # Every now and then.. save intermediate results..
                if epoch % max(n_epoch_save, 1) == 0:
                    epoch_temp_weight_name = str(n_epoch_save).zfill(3) + "_" + self.name_temp_weights
                    dir_temp_weights = os.path.join(self.config_param['dir']['doutput'], epoch_temp_weight_name)
                    torch.save(self.temp_weights, dir_temp_weights)
                    plt.close('all')
                    self.save_model(plot_name='intermediate')

            except KeyboardInterrupt:
                print('\t\t Keyboard interrupt ')
                break

        if breakdown_counter > self.breakdown_limit:
            print('Training \t We are not advancing fast enough. Broke out of training loop')
        elif epoch >= n_epoch:
            print('Training \t Completed all epochs')
        else:
            print('Training \t Increase in loss.. break down')

        self.history_dict['train_loss'] = epoch_loss_curve  # batch_loss_curve
        self.history_dict['loss_value'] = loss_value_curve  # first loss curve.
        self.history_dict['loss_value_additional'] = loss_additional_curve  # second loss curve
        self.history_dict['val_loss'] = val_loss_curve

        return self.history_dict

    def train_model_hsic(self):

        # :param kwargs: if you need to pass more arguments to the compiler
        if self.model_obj is None:
            warnings.warn('Please run self.get_model() first')

        if self.optimizer_obj is None:
            warnings.warn('Please define an optimizer_obj first')

        if self.debug:
            print('INFO - EXEC: \t Using optimizer ', self.optimizer_obj)
            print('INFO - EXEC: \t Using loss ', self.loss_obj)

        min_epoch_loss = 9999
        breakdown_counter = 0

        epoch_loss_curve = []
        val_loss_curve = []
        self.model_obj.train()  # IMPORTANT
        n_epoch = self.config_param['model']['n_epoch']
        n_extra_iteration = self.config_param['model'].get('n_extra_iteration', 0)
        return_gradients = self.config_param['model'].get('return_gradients', False)
        if return_gradients:
            self.history_dict.setdefault('overal_grad', [])
            self.history_dict.setdefault('overal_param', [])

        epoch = 0
        while epoch < n_epoch and breakdown_counter < self.breakdown_limit:
            try:
                print(f"Epoch {epoch + 1}/{n_epoch} ...")
                print(f'Breakdown counter {breakdown_counter}')

                # Train
                epoch_loss = 0
                if self.debug:
                    print('EXEC - TRAIN: Start training process')

                for container in self.train_loader:
                    if self.debug:
                        print('EXEC - TRAIN: First batch loaded')

                    if self.masked or self.loss_config['run'].get('masked', False) or self.loss_additional_config['run'].get('masked', False):
                        X, y, mask = container
                        mask = mask.to(self.device)
                    else:
                        X, y = container

                    torch_input, torch_target = X.to(self.device), y.to(self.device)

                    if self.debug:
                        print('EXEC - TRAIN: Put to device ')

                    # # # # Calculate the output AND the hidden layers
                    # torch_pred, hiddens = self.model_obj(torch_input)

                    # Copy the input as preliminary output
                    torch_pred = torch_input.copy()
                    res = htmisc.get_all_children(self.model_obj)
                    hiddens = []
                    with torch.no_grad():
                        for i_layer in res:
                            torch_pred = i_layer(torch_pred)
                            hiddens.append(torch_pred)
                    # # # #

                    if self.masked:
                        torch_pred = torch_pred * mask

                    if self.debug:
                        print('EXEC - TRAIN: Batch prediction done')

                    # # # # #
                    # # # # #  Renovate the loss object..
                    # # # # #
                    if self.debug:
                        print('EXEC - TRAIN: Calculate hidden states and gradients HSIC')

                    # Copy paste form https://github.com/choasma/HSIC-bottleneck/blob/master/source/hsicbt/core/train_hsic.py
                    # h_target = torch_pred.view(-1, 1)
                    # h_target = misc.to_categorical(h_target, num_classes=10).float()
                    # I believe that the target should, in my case, be shaped in this way.
                    h_target = torch_input.view(-1, np.prod(torch_target.size()[1:]))
                    h_data = torch_input.view(-1, np.prod(torch_input.size()[1:]))

                    for i in range(len(hiddens)):
                        # torch_pred, hiddens = self.model_obj(torch_input)
                        # TODO fix the torch_input redundancy
                        res = htmisc.get_all_children(self.model_obj)
                        hiddens = []
                        with torch.no_grad():
                            for i_layer in res:
                                torch_input = i_layer(torch_input)
                                hiddens.append(torch_input)

                        # #  get layer parameters
                        # TODO select only one layer at a time here..
                        # (Or atleastr two consecutive..
                        param_names, params = zip(*self.model_obj.named_parameters())
                        temp_optimizer = torch.optim.SGD(params, **self.config_param['optimizer']['config'])
                        temp_optimizer.zero_grad()
                        if len(hiddens[i].size()) > 2:  # If the amount of channels is too large...
                            hiddens[i] = hiddens[i].view(-1, np.prod(hiddens[i].size()[1:]))

                        sigma = 0.1  # Some parameter... set it to .. a "random" value
                        hy_l = self.hsic_normalized_cca(hiddens[i], h_target, sigma=sigma)
                        hx_l = self.hsic_normalized_cca(hiddens[i], h_data, sigma=sigma)

                        lambda_y = 0.1  # Some parameter... set it to .. a "random" value
                        loss = hx_l - lambda_y * hy_l
                        loss.backward()
                        temp_optimizer.step()

                    if self.debug:
                        print('EXEC - TRAIN: Loss calculated')

                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    if self.debug:
                        print('EXEC - TRAIN: Perform gradient step')

                    # batch_loss_curve.append(loss.item())
                    epoch_loss += loss.item()
                    self.scheduler_model_obj.step()

                    if return_gradients:
                        list_children = htmisc.get_all_children(self.model_obj, [])
                        sel_layer_name, sel_layer_param = htmisc.get_all_parameters(list_children)

                        grad_level = htmisc.get_grad_layers(sel_layer_param, sel_layer_name)
                        param_level = htmisc.get_param_layers(sel_layer_param, sel_layer_name)
                        grad_name, grad_array = zip(*grad_level)
                        param_name, param_array = zip(*param_level)
                        grad_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in grad_array]
                        param_per_layer = [(float(x.min()), float(x.mean()), float(x.max())) for x in param_array]

                        self.history_dict['overal_grad'].append(grad_per_layer)
                        self.history_dict['overal_param'].append(param_per_layer)

                val_loss = self.validate_model()
                val_loss_value = val_loss.data.tolist()
                val_loss_curve.append(val_loss_value)

                # Average over amount of batches..
                epoch_loss = epoch_loss / self.train_loader.__len__()
                epoch_loss_curve.append(epoch_loss)
                print(f'Training \t Average loss over batch: {float(epoch_loss): .4f}')

                if epoch_loss < min_epoch_loss:
                    if self.debug:
                        print(f'Training \t Old minimum loss {float(min_epoch_loss): .4f}')
                        print(f'Training \t New minimum loss {float(epoch_loss): .4f}')
                    min_epoch_loss = epoch_loss
                    self.temp_weights = self.model_obj.state_dict()

                # We want to have a decrease in loss...
                if epoch_loss != 0:
                    # For all the non-zero losses...
                    temp_curve = [x for x in val_loss_curve if x != 0]
                    historic_loss = np.mean(temp_curve[-(self.memory_time + self.memory_length): -self.memory_time])
                    current_loss = np.mean(temp_curve[-self.memory_length:])
                    criterion = historic_loss - current_loss
                    print(f'validation loss criterion {criterion}\n'
                          f'historic loss {historic_loss} \n'
                          f'validation loss {val_loss_value}')
                    # If the current loss gets larger than the historic loss...
                    if criterion < 0:
                        breakdown_counter += 1
                        if self.debug:
                            print('INFO - EXEC: \t Not improved, counter: ', breakdown_counter, self.breakdown_limit)
                    else:
                        breakdown_counter -= 1
                        breakdown_counter = max(breakdown_counter, 0)
                        if self.debug:
                            print('INFO - EXEC: \t Improved! Compared to', historic_loss, current_loss)

                # prev_loss = epoch_loss
                epoch += 1

                # Every now and then.. save intermediate results..
                if epoch % max(int(0.10 * n_epoch), 1) == 0:
                    dir_temp_weights = os.path.join(self.config_param['dir']['doutput'], self.name_temp_weights)
                    torch.save(self.temp_weights, dir_temp_weights)
                    plt.close('all')
                    self.save_model(plot_name='intermediate')
            except KeyboardInterrupt:
                print('\t\t Keyboard interrupt ')
                break

        if breakdown_counter > self.breakdown_limit:
            print('INFO - EXEC: \t We are not advancing fast enough. Broke out of training loop')
        elif epoch >= n_epoch:
            print('INFO - EXEC: \t Completed all epochs')
        else:
            print('INFO - EXEC: \t Increase in loss.. break down')

        self.history_dict['train_loss'] = epoch_loss_curve  # batch_loss_curve
        self.history_dict['val_loss'] = val_loss_curve

        return self.history_dict

    def validate_model(self, epoch=None):
        self.model_obj.eval()  # IMPORTANT

        loss = 0
        with torch.no_grad():  # IMPORTANT
            for container in self.val_loader:
                if self.masked or self.loss_config['run'].get('masked', False) or self.loss_additional_config['run'].get('masked', False):
                    X = container['input']
                    y = container['target']
                    mask = container.get('mask', None)
                    if mask is None:
                        mask = container.get('mask_target', None)
                    if mask is None:
                        print('Sorry, cant find mask or mask_target')
                    else:
                        mask = mask.to(self.device)
                else:
                    X = container['input']
                    y = container['target']

                X = X.to(self.device)
                y = y.to(self.device)

                if self.debug:
                    print('Validate \t Put container content to device ')

                if self.trained_model_config.get('status', False):
                    X = self.trained_modelrun_obj.model_obj(X)

                y_pred = self.model_obj(X)

                if self.masked:
                    y_pred = y_pred * mask
                    y = y * mask

                # Ferry ugly...... only on loss for now.. maybe later for additional one as wel...
                if self.loss_config['run'].get('masked', False):
                    if self.loss_config['run'].get('target_is_difference', False):
                        loss += self.loss_obj(y_pred, y - X, mask=mask) * self.lambda_loss(999)
                    elif self.loss_config['run'].get('input_to_loss', False):
                        loss += self.loss_obj(y_pred, y, mask=mask, input=X) * self.lambda_loss(999)
                    else:
                        loss += self.loss_obj(y_pred, y, mask=mask) * self.lambda_loss(999)
                else:
                    if self.loss_config['run'].get('target_is_difference', False):
                        loss += self.loss_obj(y_pred, y - X) * self.lambda_loss(999)
                    elif self.loss_config['run'].get('input_to_loss', False):
                        loss += self.loss_obj(y_pred, y, x_input=X) * self.lambda_loss(999)
                    else:
                        loss += self.loss_obj(y_pred, y) * self.lambda_loss(999)

                if self.loss_additional_obj is not None:
                    if self.loss_additional_config['run'].get('masked', False):
                        temp_loss = self.loss_additional_obj(y_pred, y, mask=mask) * self.lambda_loss_additional(999)
                    elif self.loss_additional_config['run'].get('input_to_loss', False):
                        temp_loss = self.loss_additional_obj(y_pred, y, x_input=X) * self.lambda_loss_additional(999)
                    elif self.loss_additional_config['run'].get('target_is_difference', False):
                        temp_loss = self.loss_additional_obj(y_pred, y - X) * self.lambda_loss_additional(999)
                    else:
                        temp_loss = self.loss_additional_obj(y_pred, y) * self.lambda_loss_additional(999)

                    print(f'Loss component {loss}, {temp_loss}')
                    loss += temp_loss

            loss = loss / len(self.val_loader)

        self.model_obj.train()  # We want to go back to the normal routine..

        return loss.item()

    @abc.abstractmethod
    def get_data_generators(self, indicator_train):
        # This one may vary over implementations
        raise NotImplementedError

    def hsic_normalized_cca(self, x, y, sigma, use_cuda=True, to_numpy=True):
        """
        Used for HSIC
        :param x:
        :param y:
        :param sigma:
        :param use_cuda:
        :param to_numpy:
        :return:
        """
        m = int(x.size()[0])
        Kxc = self.kernelmat(x, sigma=sigma)
        Kyc = self.kernelmat(y, sigma=sigma)

        epsilon = 1E-5
        K_I = torch.eye(m)
        Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
        Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
        Rx = (Kxc.mm(Kxc_i))
        Ry = (Kyc.mm(Kyc_i))
        Pxy = torch.sum(torch.mul(Rx, Ry.t()))

        return Pxy

    def kernelmat(self, X, sigma):
        """
        kernel matrix baker
        Used for HSIC
        """
        m = int(X.size()[0])
        dim = int(X.size()[1]) * 1.0
        H = torch.eye(m) - (1. / m) * torch.ones([m, m])
        Dxx = self.distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = self.sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError(
                    "Unstable sigma {} with maximum/minimum input ({},{})".format(
                        sx, torch.max(X), torch.min(X)))

        Kxc = torch.mm(Kx, H)

        return Kxc

    # # #
    def distmat(self, X):
        """
        distance matrix
        Used for HSIC
        """
        r = torch.sum(X * X, 1)
        r = r.view([-1, 1])
        a = torch.mm(X, torch.transpose(X, 0, 1))
        D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
        D = torch.abs(D)
        return D

    def sigma_estimation(self, X, Y):
        """
        sigma from median distance
        Used for HSIC
        """
        D = self.distmat(torch.cat([X, Y]))
        D = D.detach().cpu().numpy()
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            med = np.mean(Tri)
        if med < 1E-2:
            med = 1E-2
        return med

    @staticmethod
    def corrcoef(x):
        # Used for xCNN layer correction
        mean_x = torch.mean(x, dim=1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c / stddev[:, None]
        c = c / stddev[None, :]
        c = torch.clamp(c, -1.0, 1.0)
        return c
