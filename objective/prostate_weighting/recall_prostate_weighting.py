import objective.prostate_weighting.executor_prostate_weighting as executor
from objective.recall_base import RecallBase
import torch
import os
import helper.misc as hmisc


class RecallProstateWeighting(RecallBase):
    def get_model_object(self, config_file=None, inference=True, load_model_only=True):
        # I dont want it to be initialized when calling the Recall
        # So I can do it here....

        # config_param = hmisc.convert_remote2local_dict(model_path, path_prefix="")
        # with open(os.path.join(model_path, name), 'r') as f:
        #     temp_text = f.read()
        #     config_model = json.loads(temp_text)

        decision_obj = executor.DecisionMakerProstateWeighting(config_file=config_file, debug=False,
                                                               load_model_only=load_model_only, inference=inference, device=None)  # ==>>
        # model_choice = decision_obj.config_param['model']['model_choice']
        modelrun_obj = decision_obj.decision_maker()
        modelrun_obj.device = torch.device('cpu')
        #
        # # Why do I need to do this again...?
        # Was this needed to try a different model from the config file...??
        # if model_choice == "regular":
        #     network_choice = hmisc.get_nested(modelrun_obj.config_param, ['model', 'config_regular', 'network_choice'])
        #     config_model = hmisc.get_nested(modelrun_obj.config_param, ['model', f'config_{network_choice}'])
        #     modelrun_obj.model_obj = modelrun_obj.get_model(config_model=config_model, model_choice=network_choice)
        # elif model_choice == "gan":
        #     network_choice = hmisc.get_nested(modelrun_obj.config_param, ['model', 'config_gan', 'network_choice'])
        #     config_model = hmisc.get_nested(modelrun_obj.config_param, ['model', f'config_{network_choice}'])
        #     modelrun_obj.generator = modelrun_obj.get_model(config_model=config_model, model_choice=network_choice)
        # elif model_choice == "cyclegan":
        #     print("Cycle gan method is not implemented yet....")
        #     pass
        # else:
        #     print("Unknown model choice. Please select regular/gan/cyclegan")
        #     modelrun_obj = None

        modelrun_obj.load_weights()
        return modelrun_obj

    def get_prediction(self):
        # With this we can re-do some of the caluclations and check how we can get the
        # best post porcessed output... because that is currently lacking, and I think it is mostly visual.
        for config_name, i_config in self.mult_dict.items():
            print('Dealing with config ', i_config)
            full_model_path = os.path.join(self.model_path, config_name)
            i_config['dir']['doutput'] = full_model_path
            modelrun_obj = self.get_model_object(i_config)

            if modelrun_obj.model_obj:
                modelrun_obj.model_obj.eval()  # IMPORTANT
            else:
                modelrun_obj.generator.eval()

            counter = 0
            with torch.no_grad():  # IMPORTANT
                for container in modelrun_obj.test_loader:
                    print('Starting with prediction number ', counter)
                    break
                    X, y, y_pred, mask = self.make_prediction(modelrun_obj, container)
                    break

            return X, y, y_pred, mask


if __name__ == "__main__":
    config_file = '/local_scratch/sharreve/model_run/prostate_weighting'
    recal_obj = RecallProstateWeighting(model_path=config_file, config_name='config_run.json')
    X, y, y_pred, mask = recal_obj.get_prediction()