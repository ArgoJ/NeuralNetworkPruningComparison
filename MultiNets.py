import numpy as np
import os
import random
import matplotlib.pyplot as plt


import Helpers

from Saves import Saves
from basicFrame import getMeanMetricsFrame, getMetricFrame, getDuringTrainFrame, Metrics, MeanMetrics, InTrainEvals
from Loads import Loadings, getDimensionSaveDirectory
from DataClasses import *
from DataClassesJSON import ConfigData
from Helpers import (
    TimeEstimation, 
    printEvals, 
    printMetrics, 
    getMeanFromRuns, 
    calcHyperparams,
    get_total_nodes,
)
from NeuralNetwork import Model, get_features, batchTrain, evalModel
from CustomExceptions import DimensionException

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from PlotScripts.Plots2d import Plots2d
from PlotScripts.Plots3d import Plots3d




def multiNets(
    config: ConfigData,
    data_cls: Data2d | Data3d, 
    plots_cls: Plots2d | Plots3d,
    print_model = False,
    print_train_log = False,
    print_evals_b = False, 
    print_metrics_b = False, 
    print_mean_metrics_b = False,
    print_models_and_runs = False
):
    """Initialises, trains and evaluate models with different configurations and 
    random training starting points. Also plots the model data, model errors and 
    the metrics with every configuration in it. 

    INPUTS
    ------
        ``config`` is the configuration file of the models of type ConfigData. 

        ``data_cl`` is an instance of Data2d or Data3d with configurations. 

        ``plots_cl`` is an instance of Data2d or Data3d with configurations. 

        ``print_evals_b`` is True if the evals should be printed of every trained model.

        ``print_metrics_b`` is True if the metrics frame should be printed at the end.

        ``print_mean_metrics_b`` is True if the mean metrics frame should be printed at the end.

    RETURN
    ------
        ``models`` is a list of trained models. 

        ``model_figures`` is filled with figures of loss, R² and error of original to model data.
        
        ``metrics`` is a frame with all the metrics for train and test data. 

        ``mean_metrics`` is a frame with all the mean metrics for train and test data. 
        
        ``hyper_frame`` is the hyperparameter frame of multi networks of type DataFrame.
    """
    # GET DATA
    train_data = data_cls.getTrainData()
    val_data = data_cls.getValidationTensor()
    test_data = data_cls.getTestData()


    ## MULTI NET LOOP WITH FRAMES
    hyper_frame = calcHyperparams(config)
    total_nodes = get_total_nodes(hyper_frame)
    metrics = getMetricFrame(
        total_nodes,
        config.runs
    )
    in_train_evals = getDuringTrainFrame(
        total_nodes,
        config.runs,
        config.epochs
    )

    models = Models([] for _ in range(config.networks))
    

    print_something = print_models_and_runs or print_mean_metrics_b or print_evals_b or print_model or print_metrics_b
    if not print_something:
        print('Models training progress')
        print('|' + '='*100 + '|')
    times = TimeEstimation(config) 


    for network in range(config.networks):
        if print_models_and_runs:
            print('='*100 + '\n' + f'Model {network+1} of {config.networks} models')
        
        # set hyperparams for net
        hyperparams = hyper_frame.iloc[network]
        layer = hyperparams.loc['layer']
        nodes = hyperparams.loc['nodes']
        features = get_features(
            config.degree, config.inputs, layer, nodes, config.outputs
        )

        for run_index in range(config.runs):
            if print_models_and_runs:
                print('-'*100 + '\n' + f'Run {run_index+1} of {config.runs}')
            
            
            ## INIT DATALOADER
            train_loader = data_cls.getDataLoader(
                train_data, 
                config.batch_size, 
                loader_reprod=config.loader_reprod,
                run=run_index
            )

            ## INIT MODEL
            model = Model(features)
            if Helpers.use_cuda:
                model = model.cuda()
            if print_model:
                print(model)


            ## TRAIN MODEL
            # train_time = perf_counter()
            model, net_during_train_evals = batchTrain(
                model, 
                train_loader, 
                val_data, 
                config.epochs, 
                config.learning_rate,
                data_cls.getTrueDataSize()[0],
                print_log=print_train_log
            )
            # print(f'training time: {perf_counter() - train_time:.2f}')
            models[network].append((features, model))

            frame_idx = (network*config.runs + run_index)*config.epochs
            in_train_evals.iloc[frame_idx : frame_idx + config.epochs] = net_during_train_evals


            ## EVALUATE MODEL
            # evaluate model with train
            eval_train = evalModel(model, train_data)
            # evaluate model with test 
            eval_test = evalModel(model, test_data)
            # fill frames
            metrics.iloc[run_index + network*config.runs] = np.append(eval_train, eval_test)
            # print evals
            if print_evals_b:
                printEvals(eval_train, eval_test)

            ## REST TIME ESTIMATION
            times.calcTimeEstimation(
                run_index, network, time_advertised=print_something
            )

        # print all frames 
        if print_metrics_b:
            net_change_hp = metrics.index.get_level_values(0).unique()
            metrics_run_indexing = metrics.index.isin([net_change_hp[network]], level=0)
            printMetrics(metrics.iloc[metrics_run_indexing], f'Metrics for one model:')
        
    # generate mean metrics
    mean_metrics = getMeanFromRuns(metrics, getMeanMetricsFrame)  

    # print mean metrics
    if print_mean_metrics_b:
        printMetrics(mean_metrics, 'Mean Metrics:')

    output = [
        hyper_frame,
        models, 
        metrics, 
        mean_metrics,
        in_train_evals
    ]
    if config.make_plots:
        try:
            rand_run_idx = random.randint(0, config.runs - 1)
            one_run_models = Models([run_models[rand_run_idx]] for run_models in models)
            output.extend(
                plotMultiNets(
                    config,
                    data_cls,
                    plots_cls,
                    one_run_models,
                    metrics,
                    mean_metrics,
                    in_train_evals,
                    rand_run_idx=rand_run_idx
                )
            )
        except:
            print('Error in plotgeneration!')
    return output




def plotMultiNets(
    config: ConfigData,
    data_cls: Data2d | Data3d, 
    plots_cls: Plots2d | Plots3d,
    models: Models,
    metrics: Metrics,
    mean_metrics: MeanMetrics,
    in_train_evals: InTrainEvals,
    rand_run_idx = None
):
    """
    INPUTS
    ------
        ``config`` is the configuration file of the models of type ConfigData. 

        ``data_cl`` is an instance of Data2d or Data3d with configurations. 

        ``plots_cl`` is an instance of Data2d or Data3d with configurations. 

        ``models`` 

        ``metrics``

        ``mean_metrics``

        ``during_train_evals``

    RETURN
    ------

    """
    original_data = data_cls.getOriginalData()
    train_data = data_cls.getTrainData()

    ## ORIGINAL PLOT
    fig_original = plots_cls.plotOrigFun(original_data)

    ## METRIC PLOTS
    if rand_run_idx is None:
        metrics_figures = plots_cls.plotAllRunMetrics(metrics)
    else:
        metr = metrics.xs(key=rand_run_idx+1, axis=0, level='run')
        metrics_figures = MetricFigures([plots_cls.plotMetrics(metr)])
        

    fig_metrics = plots_cls.plotMeanMetrics(mean_metrics)
    metrics_figures.append(fig_metrics)

    model_figures = ModelFigures([] for _ in range(config.networks))

    for network, run_models in enumerate(models):
        for run_index, (features, model) in enumerate(run_models):
            frame_idx = (network*config.runs + (run_index if rand_run_idx is None else rand_run_idx))*config.epochs

            if config.inputs==1:
                model_data = data_cls.getModelData(model, original_data)
                fig_model = plots_cls.plotModelData(
                    original_data,
                    model_data,
                    data_cls.toNumpy(train_data),
                    in_train_evals.iloc[frame_idx : frame_idx + config.epochs]
                )
            elif config.inputs==2:
                error_data = data_cls.getError(model, original_data)
                fig_model = plots_cls.plotModelData(
                    error_data, 
                    data_cls.reshapeForPlot(train_data),
                    in_train_evals.iloc[frame_idx : frame_idx + config.epochs]
                )
            model_figures[network].append(fig_model)

    return(
        fig_original,
        model_figures,
        metrics_figures
    )




def plotAllMultiNets(
    plots_cls: Plots2d | Plots3d,
    dimension,
    one_run_modelPlot = False,
    save_directly = False
):
    base_directory = getDimensionSaveDirectory(dimension)
    folders = os.listdir(base_directory)

    for folder in folders:
        if False and ('02_16_20_25'==folder or '02_16_17_41'==folder or '02_16_13_03'==folder or '02_16_14_31'==folder):
            del folder
            continue

        print('='*100 + '\n' + f'{folder} Folder')

        loadings_cls = Loadings(os.path.join(base_directory, folder))
        config = loadings_cls.loadBaseConfig()
        if config.inputs==1:
            # 2D stuff
            data_cls = Data2d(config)
        elif config.inputs==2:
            # 3D stuff
            data_cls = Data3d(config)
        else:
            raise(DimensionException(config.inputs))

        rand_run_idx = 0 # random.randint(0, config.runs - 1) if one_run_modelPlot else None
        models = loadings_cls.loadAllModels(config, rand_run_idx=rand_run_idx)
        metrics, mean_metrics = loadings_cls.loadMetrics()
        in_train_evals = loadings_cls.loadInTrainEvals()
        hyper_frame = calcHyperparams(config)

        plots = plotMultiNets(
            config,
            data_cls,
            plots_cls,
            models,
            metrics,
            mean_metrics,
            in_train_evals,
            rand_run_idx=rand_run_idx
        )

        if not save_directly:
            plt.show()
            save = input('To save data, type in \'y\'.\n')
        else:
            save = 'y'
        if 'y' in save:
            saves_cls = Saves(base_dimension=dimension)
            saves_cls.saveMultiNetPlotsAndIllus((folder, hyper_frame, *plots),)
        plt.close()
