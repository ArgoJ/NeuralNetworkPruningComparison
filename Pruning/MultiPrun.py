import numpy as np
import os
import random
import matplotlib.pyplot as plt


import Helpers

from Saves import Saves
from basicFrame import (
    Metrics, 
    MeanMetrics, 
    Sparsities,
    IterMetrics, 
    IterMeanMetrics, 
    IterSparsity, 
    getMetricFrame, 
    getSparsitiesFrame,
    getIterMetricsFrame, 
    getIterSparsityFrame, 
    getMeanMetricsFrame
)
from Helpers import (
    TimeEstimation,
    printEvals, 
    printMetrics, 
    getMeanFromRuns, 
    calcHyperparams, 
    comparissonIndexes, 
    getSlicedMetrics, 
    getIterMeanMetrics, 
    add_new_index_chp, 
    get_total_nodes,
    get_new_total_nodes,
)
from NeuralNetwork import evalModel
from DataClasses import *
from DataClassesJSON import *
from Constants import MEAN_NAMES
from Loads import Loadings, checkAllMetricsLoaded

from PlotScripts.Plots2d import Plots2d
from PlotScripts.Plots3d import Plots3d

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Pruning.Const_Prun_Types import PRUN_TYPES
from Pruning.Const_Prun_Methods import PRUN_METHODS
from Pruning.PrunHelpers import removePruningReparamitrizations, getPrunConfig, removeNodes, getSparsity
from Pruning.PruningMethods.MagnitudePruning import structured_prev_params, structured_next_params

from Illustrations.ModelGraph import makeGraph
from Illustrations.ModelDraws import makeDraws





def loadModelAndPrune( 
    loadings_cls: Loadings, 
    data_cls: Data2d | Data3d, 
    plots_cls: Plots2d | Plots3d,
    loaded_config: ConfigData, 
    method_config: PrunConfig | MethodPrunConfig,
    print_model = False,
    print_evals_b = False, 
    print_metrics_b = False, 
    print_mean_metrics_b = False,
    print_models_and_runs = False
):
    """

    INPUTS
    ------
        ``loadings`` is the Loadings instance to load models and stuff of a specific fodler. 

        ``plots_cls`` is the Plots instance with the plot funs.  

        ``data_cls`` is the Data instance with generated original, train, validation and test data. 

        ``loaded_config`` is the configuration file of the loaded models.  

        ``model_config`` is the configuration file of the prune models. 

        ``method_config`` is the configuration file of the pruning method for the models. 

    RETURN
    ------
        ``data_list`` 
    """
    # hyperparameters
    model_config = getPrunConfig(loaded_config, method_config)
    prun_hyperparams = calcHyperparams(model_config)
    total_nodes = get_total_nodes(prun_hyperparams)

    # pruned models
    prun_models = Models([] for _ in range(model_config.networks))

    if method_config.remove_nodes:
        prun_model_not_rem_nodes = ModelsNotRemovedNodes([] for _ in range(model_config.networks))
    
    # prun metrics
    prun_metrics = getMetricFrame(
        total_nodes,
        model_config.runs,
    )
    prun_sparsities = getSparsitiesFrame(
        total_nodes,
        model_config.runs,
    )
    
    # iter stuff
    if type(method_config) is MethodPrunConfig and method_config.iterations > 1 and method_config.method!='syn_flow':
        iter_metrics = getIterMetricsFrame(
            total_nodes,
            model_config.runs, 
            method_config.iterations
        )
        iter_sparsity = getIterSparsityFrame(
            total_nodes,
            model_config.runs, 
            method_config.iterations
        )

    # new features
    new_total_nodes = np.zeros((model_config.networks, model_config.runs), dtype=np.number)

    # network start and end idx for loaded metrics
    network_idxs = comparissonIndexes(loaded_config, model_config)

    # if someting is printed 
    print_something = print_models_and_runs or print_mean_metrics_b or print_evals_b or print_model or print_metrics_b
    if not print_something:
        print('Pruning progress')
        print('|' + '='*100 + '|')

    # for time estimation
    times = TimeEstimation(model_config)
    
    for network in range(model_config.networks):
        if print_models_and_runs:
            print('='*100 + '\n' + f'Model {network+1} of {model_config.networks} models')
    

        for run_index in range(model_config.runs):
            if print_models_and_runs:
                print('-'*100 + '\n' + f'Run {run_index+1} of {model_config.runs}')

            frame_idx = run_index + network*model_config.runs

            ## LOAD MODEL
            features, model = loadings_cls.loadOriginalModel(run_index, network_idxs[network])
            if Helpers.use_cuda:
                model = model.cuda()
            if print_model:
                print(model)


            ## PRUN MODEL
            prun_model, iter_metrics_net, iter_sparsity_net = prune_model(
                model,
                method_config,
                model_config,
                data_cls,
                features,
                run_index,
                print_something
            )

            if type(method_config) is MethodPrunConfig and method_config.iterations > 1 and method_config.method!='syn_flow':
                iter_frame_idx = frame_idx*(method_config.iterations + 1)
                iter_metrics.iloc[iter_frame_idx : iter_frame_idx + method_config.iterations + 1] = iter_metrics_net
                iter_sparsity.iloc[iter_frame_idx : iter_frame_idx + method_config.iterations + 1] = iter_sparsity_net

            # remove pruning reparams
            rep_model = removePruningReparamitrizations(prun_model)
            prun_sparsities.iloc[frame_idx, 1] = getSparsity(rep_model, method_config.bias_or_weight, print_sparsity=False)
            if method_config.remove_nodes:
                prun_model_not_rem_nodes[network].append((features, rep_model))
                features, rep_model = removeNodes(features, rep_model)
                prun_sparsities.iloc[frame_idx, 0] = getSparsity(rep_model, method_config.bias_or_weight, print_sparsity=False)

            new_total_nodes[network, run_index] = get_new_total_nodes(features)
            prun_models[network].append((features, rep_model))


            ## EVALUATE MODEL
            # evaluate model with train
            eval_train = evalModel(rep_model, data_cls.getTrainData())
            # evaluate model with test 
            eval_test = evalModel(rep_model, data_cls.getTestData())
            # fill frames
            prun_metrics.iloc[frame_idx] = np.append(eval_train, eval_test)

            # print evals
            if print_evals_b:
                printEvals(eval_train, eval_test)

            ## REST TIME ESTIMATION
            # if type(method_config) is MethodPrunConfig:
            times.calcTimeEstimation(
                run_index, network, time_advertised=print_something
            )

    # mean metrics
    prun_mean_metrics = getMeanFromRuns(prun_metrics, getMeanMetricsFrame)

    # new index of metrics
    add_new_index_chp(
        prun_metrics, 
        new_total_nodes
    )
    add_new_index_chp(
        prun_mean_metrics, 
        np.tile(np.mean(new_total_nodes, axis=1), (len(MEAN_NAMES), 1)).T
    )

    # print mean metrics
    if print_mean_metrics_b:
        printMetrics(prun_mean_metrics, 'Pruned mean metrics:')

    output = [
        prun_hyperparams,
        prun_models, 
        prun_metrics, 
        prun_mean_metrics,
        prun_sparsities,
    ]

    if type(method_config) is MethodPrunConfig and method_config.iterations > 1 and method_config.method!='syn_flow':
        iter_mean_metrics = getIterMeanMetrics(iter_metrics)
        output.extend((iter_metrics, iter_mean_metrics, iter_sparsity))
    else:
        iter_metrics, iter_mean_metrics, iter_sparsity = None, None, None

    if method_config.remove_nodes:
        output.extend((prun_model_not_rem_nodes, ))


    # make plots
    if model_config.make_plots:
        try:
            rand_run_idx = random.randint(0, model_config.runs - 1)
            one_run_models = Models(
                [run_models[rand_run_idx]] for run_models in (
                    prun_model_not_rem_nodes if method_config.remove_nodes else prun_models
                )
            )
                # ... because models are just needed for illustrations here 

            output.extend(
                plotsAndIllusPruned(
                    plots_cls, 
                    loaded_config, model_config, method_config,
                    one_run_models,
                    *loadings_cls.loadMetrics(),
                    prun_metrics, prun_mean_metrics,
                    prun_sparsities,
                    iter_metrics, iter_mean_metrics,
                    iter_sparsity
                )
            )
        except:
            print('Error in plotgeneration!')
    return output




def prune_model(
    model: Model,
    method_config: MethodPrunConfig | PrunConfig,
    model_config: ConfigData,
    data_cls: Data2d | Data3d, # for PrunConfig this can be None
    features: list[int], # for PrunConfig this can be None
    run_index: int,
    print_something: bool
):
    """Calculates the mean of metric of every network over all runs. 

    INPUTS
    ------
        ``model`` is the model that should be pruned.
        
        ``method_config`` is the config of the pruning method.

        ``model_config`` is the config of the models for lr, epochs, batchsizes...

        ``data_cls`` is the data class for retraining. 
        (can be ``None`` when pruned with only MAGNITUDE PRUNING)

        ``features`` are the features of the model and is needed for reinit the model. 
        (can be ``None`` when NO REINIT is done)

        ``run_index`` is the run index for seeding the model and dataloader. 
        (can be ``None`` when pruned with only MAGNITUDE PRUNING)

        ``print_something`` is False when nothing is printed and therefore 
        only the progressbar is printed. 

    RETURN
    ------
        ``prun_model`` 

        ``eval_iter`` 

        ``iter_sparsity_net`` 
    """
    if type(method_config) is MethodPrunConfig:
        prun_model, eval_iter, iter_sparsity_net = PRUN_METHODS[method_config.method](
            model,
            method_config,
            model_config,
            data_cls,
            features,
            run_index,
            print_something
        )
    elif type(method_config) is PrunConfig:
        prun_model = PRUN_TYPES[method_config.type](
            model, 
            method_config.bias_or_weight, 
            method_config.amount
        )
        eval_iter, iter_sparsity_net = None, None

    if method_config.prun_next_params and not method_config.last_iter_npp:
        prun_model = structured_next_params(prun_model)
    if method_config.prun_prev_params:
        prun_model = structured_prev_params(prun_model)
    return prun_model, eval_iter, iter_sparsity_net




def plotsAndIllusPruned(
    plots_cls: Plots2d | Plots3d,
    loaded_config: ConfigData,
    model_config: ConfigData,
    method_config: PrunConfig | MethodPrunConfig,
    models: Models | ModelsNotRemovedNodes,
    # loaded
    loaded_metrics: Metrics,
    loaded_mean_metrics: MeanMetrics,
    # prun
    prun_metrics: Metrics,
    prun_mean_metrics: MeanMetrics,
    prun_sparsities: Sparsities,
    # iter
    iter_metrics: IterMetrics = None,
    iter_mean_metrics: IterMeanMetrics  = None,
    iter_sparsity: IterSparsity  = None
):
    """

    INPUTS
    ------
        ``model`` 

    RETURN
    ------
        ``prun_model`` 
    """
    network_idxs = comparissonIndexes(loaded_config, model_config)
    checkAllMetricsLoaded(loaded_config, model_config, loaded_metrics, prun_metrics)

    # plot mean metrics
    fig_mean_metrics = plots_cls.plotMeanMetrics(
        loaded_mean_metrics, 
        prun_mean_metrics
    )

    # plot prun metrics 1 to 1 comparisson 
    sliced_loaded_metrics = getSlicedMetrics(
        loaded_metrics, network_idxs
    )
    fig_prun_metrics = plots_cls.plotPrunMetrics(method_config, sliced_loaded_metrics, prun_metrics, prun_sparsities)
    fig_prun_metrics.insert(0, fig_mean_metrics)


    output = [
        # *make_illustrations(models),
        fig_prun_metrics
    ]

    if iter_metrics is not None and iter_mean_metrics is not None and iter_sparsity is not None: 
        figs_iter_prun = plots_cls.plotIterSparseMetrics(
            iter_mean_metrics,
            iter_sparsity.iloc[iter_sparsity.index.get_level_values(1) == model_config.runs]
        )
        output.append(figs_iter_prun)
    return output




def make_illustrations(
    models: Models
):
    """Makes the Grpahs and the Drawings for the models.

    INPUTS
    ------
        ``model_config``

        ``models`` is an instance Models filled with all models. 

    RETURN
    ------
        ``graphs``, ``draws`` are the illustrations in the instances Graphs and Draws. 
    """
    # list inits
    models_len = len(models)
    graphs = Graphs([] for _ in range(models_len))
    draws = Draws([] for _ in range(models_len))
    for network, run_models in enumerate(models):  
        for _, model in run_models:
            # make illustrations 
            graphs[network].append(makeGraph(model))
            draws[network].append(makeDraws(model))
    return graphs, draws
    



def plotAllPrunedFolders(
    loadings_cls: Loadings, 
    plots_cls: Plots2d | Plots3d,
    loaded_config: ConfigData,
    one_run = False,
    save_directly = False
):
    """

    INPUTS
    ------
        ``loadings_cls``

        ``plots_cls`` 

        ``loaded_config``

    RETURN
    ------
        ``prun_plots_illus``
    """
    folders = os.listdir(loadings_cls.getPrunDirectory())

    for folder in folders:
        if 'lottery_ticket' not in folder or 'lnp' not in folder:
            continue
        print('='*100 + '\n' + f'{folder} Method')
        
        loadings_cls.setPrunMethodDirectorys(folder)
        method_config = loadings_cls.loadMethodConfig()
        model_config = getPrunConfig(loaded_config, method_config)
        prun_hyperparams = calcHyperparams(model_config)

        rand_run_idx = 0 # np.random.choice(model_config.runs, size=1) if one_run else None
        loaded_input = [
            loadings_cls.loadAllModels_ForPrun(model_config, loaded_config, use_notRemNode_models=method_config.remove_nodes, rand_run_idx=rand_run_idx),
            *loadings_cls.loadMetrics(),
            *loadings_cls.loadPrunMetrics(),
            loadings_cls.loadPrunSparsities(),
        ]

        if type(method_config) is MethodPrunConfig and method_config.iterations > 1 and method_config.method!='syn_flow':
            loaded_input.extend((
                *loadings_cls.loadIterMetrics(), 
                loadings_cls.loadIterSparsity()
            ))

        plots_and_illus_pruned = plotsAndIllusPruned(
            plots_cls,
            loaded_config,
            model_config,
            method_config,
            *loaded_input
        )
        
        if not save_directly:
            plt.show()
            save = input('To save data, type in \'y\'.\n')
        else:
            save = 'y'
        if 'y' in save:
            saves_cls = Saves(save_directory=loadings_cls.getBaseDirectory())
            saves_cls.saveMultiPrunPlotsAndIllus((*plots_and_illus_pruned, method_config, prun_hyperparams))
        plt.close()