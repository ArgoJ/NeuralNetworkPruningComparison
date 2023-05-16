import numpy as np
import os
import itertools


import Helpers

from basicFrame import getPerformanceFrame, getMeanPerformanceFrame, Performances, MeanPerformances

from Saves import add_method_to_frame
from MultiNets import calcHyperparams
from Helpers import (
    getMeanFromRuns, 
    add_to_existing_frame, 
    find_model,
    get_total_nodes,
    TimeEstimation,
)
from Constants import PERFORMANCE_NAMES, METRIC_NAMES
from DataClassesJSON import *
from Loads import Loadings

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Pruning.PrunHelpers import getPrunConfig

from PerformanceScripts.PerformanceCheck import ModelAnalysis





def getPerformancesPruned( 
    loadings_cls: Loadings, 
    data_cls: Data2d | Data3d, 
    config: ConfigData,
    base_dirs: list[str] = [],
):  
    all_prun_folders = list(itertools.chain.from_iterable(
        [[(base_dir, folder) for folder in os.listdir(os.path.join(base_dir, 'Pruning'))] for base_dir in base_dirs]
    ))
    num_prun_folders = len(all_prun_folders)
    print(f'{num_prun_folders} pruining folders to check for performance!')
    
    
    ## PRUNED MODEL PERFORMANCES
    prun_performances: Performances = None
    prun_mean_performances: MeanPerformances = None

    time_estim_cls = TimeEstimation(config, time_rows=num_prun_folders) 


    for folder_idx, (base_dir, folder) in enumerate(all_prun_folders):
        loadings_cls.updateDirectorys(base_dir)
        loadings_cls.setPrunMethodDirectorys(folder)

        method_config = loadings_cls.loadMethodConfig()
        prun_config = getPrunConfig(config, method_config)
        prun_hyperparams = calcHyperparams(prun_config)

        np_performance = multiNetPerfs(
            loadings_cls, 
            data_cls, 
            prun_config,
            loadings_cls.getPrunMethodDirectory(),
            time_estim_cls,
            folder_idx
        )

        performance_frame = getPerformanceFrame(get_total_nodes(prun_hyperparams), config.runs, np_performance)
        mean_performance_frame = getMeanFromRuns(performance_frame, getMeanPerformanceFrame)

        indexed_prun_performances = add_method_to_frame(performance_frame, method_config)
        indexed_prun_mean_performances = add_method_to_frame(mean_performance_frame, method_config)

        prun_performances = add_to_existing_frame(prun_performances, indexed_prun_performances)
        prun_mean_performances = add_to_existing_frame(prun_mean_performances, indexed_prun_mean_performances)
    return Performances(prun_performances), MeanPerformances(prun_mean_performances)




def getPerformancesOriginal(
    loadings: Loadings, 
    data_cls: Data2d | Data3d, 
    config: ConfigData
):  
    orig_hyperparams = calcHyperparams(config)

    print('='*100 + '\n' + f'Original performance')

    time_estim_cls = TimeEstimation(config)

    np_orig_performance = multiNetPerfs(
            loadings, 
            data_cls, 
            config,
            loadings.getOrigBaseDirectory(),
            time_estim_cls,
        )
    orig_performance = getPerformanceFrame(get_total_nodes(orig_hyperparams), config.runs, np_orig_performance)
    orig_mean_performance = getMeanFromRuns(orig_performance, getMeanPerformanceFrame)
    return orig_performance, orig_mean_performance
    



def multiNetPerfs(
    loadings: Loadings, 
    data_cls: Data2d | Data3d, 
    config: ConfigData,
    model_directory: str,
    time_estim_cls: TimeEstimation,
    folder_idx: int = 0,
    print_something = False,
):
    model_path, _ = find_model(model_directory)
    np_performance = np.zeros((config.runs*config.networks, len(PERFORMANCE_NAMES) + len(METRIC_NAMES)))

    for network in range(config.networks):  
        if print_something:
            print('-'*100 + '\n' + f'Model {network+1} of {config.networks} models')  

        for run_idx in range(config.runs):
            if print_something:
                print(f'Run {run_idx+1} of {config.runs}')

            ## LOAD MODEL
            _, model = loadings.loadModel(
                run_idx, 
                network,
                model_path
            )
            if Helpers.use_cuda:
                model = model.cuda()

            index = run_idx + network*config.runs
            model_analysis = ModelAnalysis(model, data_cls)
            np_performance[index, :] = model_analysis.get_everything()

            time_estim_cls.calcTimeEstimation(run_idx, network, row_idx=folder_idx, time_advertised=print_something)
    return np_performance