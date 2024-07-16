# encoding: utf-8

import data_generator.Rx2Tx as data_gen_rx_tx
import data_generator.InhomogRemoval as data_gen_inhomog
import data_generator.UndersampledRecon as data_gen_ur
from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase
import helper.misc as hmisc
import json


class DecisionMaker(DecisionmakerBase):
    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        if 'cyclegan' == model_choice.lower():
            print('You have chosen a Cycle GAN')
            sel_executor = ExecutorRxTxCycleGAN(config_file=self.config_param, **self.kwargs)
        elif 'gan' == model_choice.lower():
            print('You have chosen a regular GAN')
            sel_executor = ExectorRxTxGAN(config_file=self.config_param, **self.kwargs)
        else:
            print('You have chosen a regular model', model_choice.lower())
            sel_executor = ExecutorRxTx(config_file=self.config_param, **self.kwargs)

        return sel_executor


class ExectorRxTxGAN(ExecutorGAN):
    def __init__(self, model_path=None, config_file=None, **kwargs):
        super().__init__(model_path=model_path, config_file=config_file, **kwargs)

    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "multiple":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_all
        elif dg_choice == "multiple_svd":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_all_svd
        elif dg_choice == "multiple_flavio":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_flavio
        else:
            def data_gen_sel(x, **kwargs):
                return x

            print('EXEC GAN \t - No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorRxTxCycleGAN(ExecutorCycleGAN):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "single":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_single
        elif dg_choice == "multiple":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_all
        elif dg_choice == "multiple_svd":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_all_svd
        else:
            def data_gen_sel(x, **kwargs):
                return x  # ppff simple fucntion

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorRxTx(ExecutorRegular):

    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "single":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_single
        elif dg_choice == "multiple":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_all
        elif dg_choice == "multiple_svd":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_all_svd
        elif dg_choice == "multiple_flavio":
            data_gen_sel = data_gen_rx_tx.DataSetSurvey2B1_flavio
        elif dg_choice == "b2b":
            data_gen_sel = data_gen_rx_tx.DataGeneratorBmin2Bplus
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator  # ppff simple fucntion

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj