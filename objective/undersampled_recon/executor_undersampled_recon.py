# encoding: utf-8

import numpy as np
import os
import helper.plot_fun as hplotf
import helper.array_transf as harray
import torch

import data_generator.UndersampledRecon as data_gen_recon
from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase
import json


class DecisionMakerRecon(DecisionmakerBase):

    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        if 'cyclegan' == model_choice.lower():
            if self.debug:
                print('You have chosen a Cycle GAN')
            sel_executor = ExecutorReconCycleGAN
        elif 'gan' == model_choice.lower():
            if self.debug:
                print('You have chosen a regular GAN')
            sel_executor = ExectorReconGAN
        else:
            if self.debug:
                print('You have chosen a regular model', model_choice.lower())
            sel_executor = ExecutorRecon

        return sel_executor(config_file=self.config_param, **self.kwargs)


class ExecutorRecon(ExecutorRegular):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "semireal":
            data_gen_sel = data_gen_recon.DataGeneratorSemireal
        elif dg_choice == "undersampled":
            data_gen_sel = data_gen_recon.DataGeneratorUndersampledRadial
        elif dg_choice == "undersampled_cart":
            data_gen_sel = data_gen_recon.DataGeneratorUndersampledCartesian
        elif dg_choice == "undersampled_proc":
            data_gen_sel = data_gen_recon.DataGeneratorUndersampledProcessed
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator  # ppff simple fucntion

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExectorReconGAN(ExecutorGAN):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice'].lower()

        if dg_choice == "semireal":
            data_gen_sel = data_gen_recon.DataGeneratorSemireal
        elif dg_choice == "undersampled":
            data_gen_sel = data_gen_recon.DataGeneratorUndersampledRadial
        elif dg_choice == "undersampled_proc":
            data_gen_sel = data_gen_recon.DataGeneratorUndersampledProcessed
        else:
            def data_gen_sel(x, **kwargs):
                return x

            print('EXEC AC GAN \t - No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj


class ExecutorReconCycleGAN(ExecutorCycleGAN):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']
        if dg_choice == "semireal":
            data_gen_sel = data_gen_recon.DataGeneratorSemireal
        elif dg_choice == "undersampled":
            data_gen_sel = data_gen_recon.DataGeneratorUndersampledRadial
        elif dg_choice == "undersampled_proc":
            data_gen_sel = data_gen_recon.DataGeneratorUndersampledProcessed
        else:
            def data_gen_sel(x, **kwargs):
                return x  # ppff simple fucntion

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj
