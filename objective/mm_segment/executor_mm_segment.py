# encoding: utf-8

from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase
import data_generator.MMSegment as data_gen


class DecisionMakerMMSegment(DecisionmakerBase):
    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        if 'cyclegan' == model_choice.lower():
            sel_executor = ExecutorMMSegmentCycleGAN
        elif 'gan' == model_choice.lower():
            sel_executor = ExecutorMMSegmentGAN
        else:
            print('You have chosen a regular model..')
            sel_executor = ExecutorMMSegment

        return sel_executor(config_file=self.config_param, **self.kwargs)


class ExecutorMMSegment(ExecutorRegular):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "segment":
            data_gen_sel = data_gen.DataGeneratorMMSegment
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorMMSegmentGAN(ExecutorGAN):
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


class ExecutorMMSegmentCycleGAN(ExecutorCycleGAN):
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
