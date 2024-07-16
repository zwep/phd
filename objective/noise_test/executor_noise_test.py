import data_generator.Default
import data_generator.Generic as data_gen_generic
from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase


class DecisionMakerNoise(DecisionmakerBase):
    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        if 'cyclegan' == model_choice.lower():
            print('You have chosen a Cycle GAN')
            sel_executor = ExecutorCycleGANNoise
        elif 'gan' == model_choice.lower():
            print('You have chosen a regular GAN')
            sel_executor = ExecutorGANNoise
        else:
            print('You have chosen a regular model', model_choice.lower())
            sel_executor = ExecutorRegularNoise

        return sel_executor(config_file=self.config_param, **self.kwargs)


class ExecutorRegularNoise(ExecutorRegular):
    def get_data_generators(self, indicator_train):
        data_gen_sel = data_generator.Default.DataGeneratorNoise

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorGANNoise(ExecutorGAN):
    def get_data_generators(self, indicator_train):
        data_gen_sel = data_generator.Default.DataGeneratorNoise

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj



class ExecutorCycleGANNoise(ExecutorCycleGAN):
    def get_data_generators(self, indicator_train):
        data_gen_sel = data_generator.Default.DataGeneratorNoise

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj

