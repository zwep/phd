# encoding: utf-8

from objective.executor_cycleGAN import ExecutorCycleGAN
from objective.executor_GAN import ExecutorGAN
from objective.executor_regular import ExecutorRegular
from objective.executor_base import DecisionmakerBase
import data_generator.ShimPrediction as data_gen
import helper.plot_fun as hplotf
import helper.misc as hmisc
import helper.array_transf as harray


class DecisionMakerShimPredicton(DecisionmakerBase):
    def decision_maker(self):
        model_choice = self.config_param['model']['model_choice']
        if 'cyclegan' == model_choice.lower():
            sel_executor = None
        elif 'gan' == model_choice.lower():
            sel_executor = None
        else:
            print('You have chosen a regular model..')
            sel_executor = ExecutorShimPredicton

        return sel_executor(config_file=self.config_param, **self.kwargs)


class ExecutorShimPredicton(ExecutorRegular):
    def get_data_generators(self, indicator_train):
        dg_choice = self.config_param['data']['generator_choice']

        if dg_choice == "shimpred":
            data_gen_sel = data_gen.DataGeneratorShimPrediction
        else:
            def data_gen_sel(unknown_data_generator, **kwargs):
                return unknown_data_generator

            print('No known data generator selected', dg_choice)

        data_gen_obj = data_gen_sel(dataset_type=indicator_train,
                                    **self.config_param['dir'],
                                    **self.config_param['data'],
                                    debug=self.debug)
        return data_gen_obj

    def postproc_output(self, torch_input, torch_target, torch_pred, sel_item):
        # Here we perform post processing on complex outputs
        # We expect the input to be of shape (batch, chan * cpx_chan ,y ,x)
        # We are going to get images that are of (batch, chan, cpx_chan, y, x)
        # So I need to fix that still..

        if self.debug:
            print('Post processing output')

        print('Example prediction', torch_pred)
        if self.debug:
            print('\t Output shape from model')
            print('\t torch input ', torch_input.shape)
            print('\t torch target ', torch_target.shape)
            print('\t torch pred ', torch_pred.shape)

        torch_input = torch_input[sel_item:sel_item+1]
        torch_target = torch_target[sel_item:sel_item + 1]
        torch_pred = torch_pred[sel_item:sel_item + 1]
        loss_container = self.loss_obj.debug_call(torch_pred, torch_target, input=torch_input)

        plot_augm = ['np.abs']
        print('Ooutput ranges of files....')

        torch_input = loss_container['result_interf']
        torch_target = loss_container['target']
        torch_pred = loss_container['result']

        if self.device == "cpu":
            torch_input = torch_input.numpy()
            torch_target = torch_target.numpy()
            torch_pred = torch_pred.numpy()
        else:
            # Then we assume CUDA..
            torch_input = torch_input.cpu().numpy()
            torch_target = torch_target.cpu().numpy()
            torch_pred = torch_pred.cpu().numpy()

        print(harray.get_minmeanmediammax(torch_input))
        print(harray.get_minmeanmediammax(torch_target))
        print(harray.get_minmeanmediammax(torch_pred))

        return torch_input, torch_target, torch_pred, plot_augm

