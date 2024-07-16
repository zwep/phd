# encoding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import helper.plot_class as hplotc
import helper.array_transf as harray
import helper.misc as hmisc
import helper.model_setting as hmodel_set
import json
from helper.misc import get_nested, set_nested, type2list
import torch
import pandas as pd

"""

"""

dir_data = '/home/bugger/Documents/model_run/test_run'


class RecallBase:
    # Collect the results of all the config..
    # Therefore it needs to load the config_run.json, such that it can recreate all the configurations.

    def __init__(self, model_path=None, config_run_file=None, config_name='config_run.json', **kwargs):

        self.debug = kwargs.get('debug')
        self.output_names = ['Prediction', 'Target', 'Difference']
        self.name_model_hist = 'model_history.json'
        # Properly process the input. Could be done better I guess
        if config_run_file is None and model_path is None:
            pass
            # print('Nothing is supplied. Continue anyway')
        if config_run_file is not None and model_path is not None:
            print('Supply only one file')
        if config_run_file is not None and model_path is None:
            if isinstance(config_run_file, dict):
                self.config_model = config_run_file
                self.model_path = self.config_model['dir']['doutput']

                # Needed to recover all the possible combinations
                packed_keys = self.config_model['packed_keys']
                self.mult_dict = hmodel_set.create_mult_dict(self.config_model, packed_keys)
                print('Recall object init. Print config_model')
                hmisc.print_dict(self.mult_dict)
            else:
                raise TypeError('Please provide a dictionary')

        if model_path is not None and config_run_file is None:
            self.model_path = model_path
            self.config_model = self.load_config(name=config_name)

            # Needed to recover all the possible combinations
            packed_keys = self.config_model['packed_keys']
            self.mult_dict = hmodel_set.create_mult_dict(self.config_model, packed_keys)
            print('Recall object init. Print config_model')
            hmisc.print_dict(self.mult_dict)

    def load_config(self, name):
        # Load the right parameters...
        with open(os.path.join(self.model_path, name), 'r') as f:
            temp = f.read()
            config_param = json.loads(temp)

        return config_param

    def _get_test_loss(self):
        # This extract the test-loss for all the configurations (excluding kspace)

        packed_keys = self.config_model['packed_keys']
        packed_keys = [type2list(x) for x in packed_keys]
        res_names = ['config'] + packed_keys + ['test loss', 'train loss']
        all_results = [res_names]

        for i_config in self.mult_dict.keys():
            temp_dict = self.mult_dict[i_config]
            res_values = [get_nested(temp_dict, x) for x in packed_keys]
            full_model_path = os.path.join(self.model_path, i_config)
            history_file_name = os.path.join(full_model_path, self.name_model_hist)

            if os.path.isfile(history_file_name):
                with open(history_file_name, 'r') as f:
                    text_obj = f.read()
                    history_obj = json.loads(text_obj)
            else:
                history_obj = {'test_loss': -1}

            # Add the train and test loss...
            train_loss_value = -1
            if len(history_obj['train_loss']):
                train_loss_value = history_obj['train_loss'][-1]

            res_values.extend([np.mean(history_obj['test_loss']), train_loss_value])
            res_values = [i_config] + res_values
            all_results.append(res_values)

        return all_results

    def write_test_result(self, name="model_losses"):
        file_data = self._get_test_loss()
        temp_df = pd.DataFrame(file_data)
        file_path = os.path.join(self.model_path, name + '.csv')
        temp_df.to_csv(file_path, index=False)
        if self.debug:
            print('INFO REC - Written test results to ', self.model_path)

    def write_figure(self, name='history_figure'):
        # Still under construction... first thing looks OKAY-ish
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')

        fig = plt.figure(figsize=(15, 10))
        ax = fig.subplots(3, 1)

        max_lr = 0
        for i_config in self.mult_dict.keys():
            full_model_path = os.path.join(self.model_path, i_config)
            model_hist_file_name = os.path.join(full_model_path, self.name_model_hist)

            if os.path.isfile(model_hist_file_name):
                with open(model_hist_file_name, 'r') as f:
                    text_obj = f.read()
                    test_loss = json.loads(text_obj)
            else:
                test_loss = {'val_loss': [-1], 'loss': [-1], 'lr': [-1]}

            if self.debug:
                hmisc.print_dict(test_loss)

            temp_val_loss = [float(x) for x in test_loss.get('val_loss', [])]
            ax[0].plot(temp_val_loss, '.-', label=i_config, alpha=0.5)
            temp_loss = [float(x) for x in test_loss.get('train_loss', [])]
            ax[1].plot(temp_loss, '-', label=i_config, alpha=0.5)
            temp_lr = [float(x) for x in test_loss.get('lr', [])]
            temp_lr = temp_lr or [-1]
            temp_max = np.max(temp_lr)
            if temp_max > max_lr:
                max_lr = temp_max
            ax[2].plot(range(len(temp_lr)), temp_lr, '-*', label=i_config, alpha=0.5)

            if self.debug:
                print('INFO - REC: \t Config ', i_config)
                print('INFO - REC: \t Val loss ', temp_val_loss)
                print('INFO - REC: \t Loss ', temp_loss)
                print('INFO - REC: \t lr ', temp_lr)

        ax[0].set_title('val_loss')
        ax[1].set_title('loss')
        ax[2].set_title('lr')
        ax[2].set_ylim([0, max_lr*1.1])

        fig.savefig(os.path.join(self.model_path, name + ".png"))
        if self.debug:
            print('INFO REC - Written figure results to ', self.model_path)

