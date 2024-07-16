import os
import abc
import json
import torch
import getpass
import helper.misc as hmisc


class PostProcBase:
    # Class that handles post processing stuff of a single configuration file
    # Used for visualization and storage of test results
    def __init__(self, executor_module, config_dict=None, config_path=None, config_name='config_run.json', **kwargs):
        self.debug = kwargs.get('debug', False)

        ind_load_model_obj = False
        if self.debug:
            print("Start loading config dict.")

        if config_dict is None and config_path is None:
            print('Nothing is supplied. Continue anyway')
        if config_dict is not None and config_path is not None:
            print('Supply only one argument: either config_dict OR config_path')
            return
        if config_dict is not None and config_path is None:
            if isinstance(config_dict, dict):
                self.config_dict = config_dict
                self.config_path = self.config_dict['dir']['doutput']
                ind_load_model_obj = True
            else:
                print('Supplied variable config_dict is not a dictionary')
                return
        if config_path is not None and config_dict is None:
            self.config_path = config_path
            self.config_dict = self.load_config(name=config_name)
            ind_load_model_obj = True

        if self.debug:
            print("Loaded config dict")
            hmisc.print_dict(self.config_dict)
            print("\nStart loading the model object")

        if ind_load_model_obj:
            self.modelrun_obj = self.get_model_object(executor=executor_module, **kwargs)
        else:
            print('Not loading a modelrun object')

        if self.debug:
            print("Modelrun object is loaded")

    def load_config(self, name):
        # Load the right parameters...
        config_path = os.path.join(self.config_path, name)
        if self.debug:
            print("Loading config path from: ", config_path)
        with open(config_path, 'r') as f:
            temp = f.read()
            config_param = json.loads(temp)

        config_param['dir']['doutput'] = self.config_path
        config_param['dir']['dtemplate'] = self.config_path
        config_param['data']['batch_size'] = 1
        if 'reload_weights_config' in config_param['model']['config_regular'].keys():
            config_param['model']['config_regular']['reload_weights_config']['status'] = False
        return config_param

    def get_model_object(self, executor, **kwargs):
        # Supply an executor module to get the appropiate model object

        load_model_only = kwargs.get('load_model_only', True)
        inference = kwargs.get('inference', False)
        debug = kwargs.get('debug', False)
        device = kwargs.get('device', None)
        decision_obj = executor.DecisionMaker(config_file=self.config_dict, debug=debug,
                                              load_model_only=load_model_only, inference=inference, device=device)  # ==>>
        modelrun_obj = decision_obj.decision_maker()
        if getpass.getuser() == 'bugger':
            # If we are doing stuff locally.. set it to cpu?
            # Not sure why, it shouldnt even find a GPU.
            modelrun_obj.device = torch.device('cpu')

        # Dit hoort toch zo te werken...??
        modelrun_obj.load_weights()

        return modelrun_obj
