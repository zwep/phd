# encoding: utf-8

import data_generator.SR as data_gen_SR
from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase
import logging


class DecisionMakerSuperRes(DecisionmakerBase):
    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        main_logger = logging.getLogger('main')
        if 'cyclegan' == model_choice.lower():
            main_logger.info('CycleGAN model selected')
            sel_executor = ExecutorSuperResCycleGAN
        elif 'gan' == model_choice.lower():
            main_logger.info('GAN model selected')
            sel_executor = ExecutorSuperResGAN
        else:
            main_logger.info('Regular model selected')
            sel_executor = ExecutorSuperRes

        return sel_executor(config_file=self.config_param, **self.kwargs)


class ExecutorSuperRes(ExecutorRegular):

    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']
        main_logger = logging.getLogger('main')
        main_logger.info(f'Following data generator was chosen: {dg_choice}')
        if dg_choice == "SR":
            data_gen_sel = data_gen_SR.DataGeneratorSR
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator

            main_logger.warning('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorSuperResGAN(ExecutorGAN):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']
        main_logger = logging.getLogger('main')
        main_logger.info(f'Following data generator was chosen: {dg_choice}')
        if dg_choice == "SR":
            data_gen_sel = data_gen_SR.DataGeneratorSR
        else:
            def data_gen_sel(x, **kwargs):
                return x

            main_logger.warning(f'No known data generator selected: {dg_choice}')

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorSuperResCycleGAN(ExecutorCycleGAN):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']
        main_logger = logging.getLogger('main')
        main_logger.info(f'Following data generator was chosen: {dg_choice}')
        if dg_choice == "SR":
            data_gen_sel = data_gen_SR.DataGeneratorSR
        else:
            def data_gen_sel(x, **kwargs):
                return x

            main_logger.warning(f'No known data generator selected: {dg_choice}')

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj
