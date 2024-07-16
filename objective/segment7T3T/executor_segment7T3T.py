# encoding: utf-8

from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase
import data_generator.Segment7T3T as data_gen


class DecisionMakerSegment7T3T(DecisionmakerBase):
    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        if 'cyclegan' == model_choice.lower():
            print('You have chosen a Cycle GAN')
            sel_executor = ExecutorSegment7T3TCycleGAN
        elif 'gan' == model_choice.lower():
            print('You have chosen a regular GAN')
            sel_executor = ExecutorSegment7T3TGAN
        elif 'regular' == model_choice.lower():
            print('You have chosen a regular model..')
            sel_executor = ExecutorSegment7T3T
        else:
            print('No model choice is found.. used: ', model_choice.lower())
            sel_executor = None

        return sel_executor(config_file=self.config_param, **self.kwargs)


class ExecutorSegment7T3T(ExecutorRegular):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "cardiac_segment":
            data_gen_sel = data_gen.DataGeneratorCardiacSegment
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorSegment7T3TGAN(ExecutorGAN):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice'].lower()

        if dg_choice == "":
            data_gen_sel = None
        else:
            def data_gen_sel(x, **kwargs):
                return x

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorSegment7T3TCycleGAN(ExecutorCycleGAN):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "":
            data_gen_sel = None
        else:
            def data_gen_sel(x, **kwargs):
                return x

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj
