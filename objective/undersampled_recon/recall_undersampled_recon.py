import torch
import helper.misc as hmisc
import objective.undersampled_recon.executor_undersampled_recon as executor
from objective.recall_base import RecallBase


class RecallUndersampled(RecallBase):
    def get_model_object(self, config_file=None, inference=False, load_model_only=True):
        # Although this method is almost identitcal for many objective funcitons..
        # The exectutor module here is not...
        # Best is to use the mult_dict['config_00'] as config file..
        # We use a config file and not a path, so that we can chance some paths etc

        decision_obj = executor.DecisionMakerRecon(config_file=config_file, debug=False,
                                                   load_model_only=load_model_only, inference=inference, device=None)  # ==>>
        modelrun_obj = decision_obj.decision_maker()
        modelrun_obj.device = torch.device('cpu')
        network_choice = hmisc.get_nested(modelrun_obj.config_param, ['model', 'config_regular', 'network_choice'])
        config_model = hmisc.get_nested(modelrun_obj.config_param, ['model', f'config_{network_choice}'])
        modelrun_obj.model_obj = modelrun_obj.get_model(config_model=config_model, model_choice=network_choice)
        modelrun_obj.load_weights()
        return modelrun_obj