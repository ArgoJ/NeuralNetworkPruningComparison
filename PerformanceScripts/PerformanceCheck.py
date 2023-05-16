import torch
import os
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from fvcore.nn import FlopCountAnalysis, parameter_count

import Helpers

from NeuralNetwork import Model, evalModel
from Helpers import createDir, getMeanFromRuns
from Constants import PERFORMANCE_NAMES, METRIC_NAMES
from basicFrame import getPerformanceFrame, getMeanPerformanceFrame, MeanPerformances, Performances
from DataClassesJSON import ConfigData

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Pruning.PrunHelpers import getSparsity



class ModelAnalysis:
    def __init__(
        self, 
        model: Model, 
        data_cls: Data2d | Data3d,
        print_something = False,
    ) -> None:
        self.curr_path = os.path.dirname(__file__)
        self.data = data_cls.getTestData()
        self.input_tensor = data_cls.getTestData()[0]
        self.model = model
        self._calc_flops()
        self._calc_params()
        self._calc_prediction_time(num_repeats=2500, num_warmups=250, print_inf_time=print_something)
        self._calc_model_size()
        _, self.weight_sparsity = getSparsity(self.model, 'weight', print_sparsity=print_something)
        _, self.bias_sparsity = getSparsity(self.model, 'bias', print_sparsity=print_something)
        self._calc_model_metrics()


    def _calc_flops(self):
        self.model.eval()
        self.flops_total = FlopCountAnalysis(self.model, self.data[0]).total()

    
    def _calc_params(self):
        self.model.eval()
        self.params_total = parameter_count(self.model)[""] # "" for total params
        

    @torch.no_grad()
    def _calc_prediction_time(self, num_repeats: int = 10000, num_warmups: int = 1000, print_inf_time = False):
        self.model.eval()

        # warmup
        if Helpers.use_cuda:
            torch.cuda.synchronize()
        for _ in range(num_warmups):
            _ = self.model.forward(self.input_tensor)

        # actual timing
        elapsed_time_s = 0
        if Helpers.use_cuda:
            torch.cuda.synchronize()
        for _ in range(num_repeats):
            start = timer()
            _ = self.model.forward(self.input_tensor)
            if Helpers.use_cuda:
                torch.cuda.synchronize()
            end = timer()
            elapsed_time_s += (end - start)
        self.pred_time = elapsed_time_s / num_repeats * 1e6  # avrage time in micro seconds

        if print_inf_time:
            print(f'inference time: {self.pred_time:>5.1f} us')


    def _calc_model_size(self):
        self.model.eval()
        buffer_directory = os.path.join(self.curr_path, 'Buffers')
        createDir(buffer_directory)
        
        buffer_path = os.path.join(buffer_directory, 'buffer_model.pt')
        # model_scripted = torch.jit.script(self.model)
        torch.save(self.model, buffer_path)
        self.model_size = os.path.getsize(buffer_path)

        if os.path.exists(buffer_path):
            os.remove(buffer_path)
        else:
            print('failed to save model!')


    def _calc_model_metrics(self):
        self.eval = evalModel(self.model, self.data)


    def get_flops(self):
        return self.flops_total

    
    def get_parameter(self):
        return self.params_total


    def get_prediction_time(self):
        return self.pred_time


    def get_model_size(self):
        return self.model_size


    def get_weight_sparsity(self):
        return self.weight_sparsity


    def get_bias_sparsity(self):
        return self.bias_sparsity

    
    def get_model_metrics(self):
        return self.eval

    
    def get_everything(self):
        return [
            self.flops_total, 
            self.model_size, 
            self.pred_time, 
            self.weight_sparsity,
            self.bias_sparsity,
            *self.eval
        ]



def getPrunedModelPerfs(
    models: list[list[tuple[list[int], Model]]], 
    data_cls: Data2d | Data3d,
    model_config: ConfigData,
    changed_hyperparam: pd.Series
) -> tuple[Performances, MeanPerformances]:
    """

    INPUTS
    ------
        ``models`` 

        ``data_cls`` is the Data instance with generated original, train, validation and test data. 

    RETURN
    ------
        ``performance_frame`` is a list of lists with the performances of every model. 
    """
    np_performance = np.zeros((model_config.runs*model_config.networks, len(PERFORMANCE_NAMES) + len(METRIC_NAMES)))
    for network, model_one_config in enumerate(models):
        for run_index, (_, model) in enumerate(model_one_config):
            index = run_index + network*model_config.runs
            model_analysis = ModelAnalysis(model, data_cls)
            np_performance[index, :] = model_analysis.get_everything()

    performance_frame = getPerformanceFrame(changed_hyperparam, model_config.runs, np_performance)
    print(f'performance:\n{performance_frame}')
    mean_performance_frame = getMeanFromRuns(performance_frame, getMeanPerformanceFrame)
    print(f'mean performance:\n{mean_performance_frame}')
    return performance_frame, mean_performance_frame