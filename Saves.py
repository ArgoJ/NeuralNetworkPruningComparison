import torch
import pandas as pd
import os
import json
import pickle
import tikzplotlib as tikzplt
import matplotlib.pyplot as plt

from datetime import datetime
from copy import deepcopy


from DataClasses import *
from DataClassesJSON import *
from Constants import *
from basicFrame import (
    HyperFrame, 
    Metrics, 
    MeanMetrics, 
    Sparsities,
    Performances, 
    MeanPerformances, 
    IterMetrics, 
    IterSparsity, 
    IterMeanMetrics, 
    InTrainEvals,
)
from Helpers import (
    add_Index_to_MultiIndex, 
    createDir, 
    createTikzDir,
    add_to_existing_frame_file, 
    rename_existing_dir,
)



class Saves:
    """Instance to save figures and models."""
    def __init__(
        self, 
        **kwargs
    ) -> None:
        """Initialises an instance of Saves where the configurations are stored and 
        the folder the files should be saved in is generated. 

        INPUTS
        ------
            ``save_directory`` is the given path as a base directory.

            ``dimension`` is the dimension folder where a new folder of type 
            '%m_%d_%H_%M' (e.g. 11_17_23_43) is used as a base directory.

            ``base_dimension`` is the basic dimension folder as a base directory.
        """
        curr_path = os.path.dirname(os.path.dirname(__file__))

        for kw, arg in kwargs.items():
            if kw=='save_directory':
                self.directory = arg
            elif kw=='dimension':
                time_now = datetime.now()
                time_str = time_now.strftime('%m_%d_%H_%M')
                self.directory = os.path.join(curr_path, f'{arg}D_Saves', time_str) 
                self.directory = rename_existing_dir(self.directory)
            elif kw=='base_dimension':
                self.directory = os.path.join(curr_path, f'{arg}D_Saves')
            else:
                raise(ValueError('No directory keyword given!'))
            
            ## ://TODO make it out of loop 
            if kw=='print_savings':
                self.print_savings = arg
            else:
                self.print_savings = False

        self.orig_directory = deepcopy(self.directory)
        createDir(self.directory)

        self.updateDirectorys(self.orig_directory)


    def updateDirectorys(self, base_folder: str):
        self.frames_folder = os.path.join(base_folder, 'Frames')

        plots_folder = os.path.join(base_folder, 'Plots')
        self.metrics_plots_folder = os.path.join(plots_folder, 'MetricsPlots')
        self.models_plots_folder = os.path.join(plots_folder, 'ModelPlots')
        self.performance_plots_folder = os.path.join(plots_folder, 'PerformancePlots')
        self.sparsity_folder = os.path.join(plots_folder, 'SparsityPlots')

        illus_folder = os.path.join(base_folder, 'Illustrations')
        self.graphs_folder = os.path.join(illus_folder, 'Graphs')
        self.draws_folder = os.path.join(illus_folder, 'Draws')


    def get_save_directory(self):
        return self.orig_directory

        

    #====================================================================================================
    # SAVE EVERYTHING
    #====================================================================================================
    def saveEverythingUnprun(
        self,
        *args,
    ):
        """Saves every given argument at the dependencie of what class it is. 
        If the hyperframe is not in the arguments, the code gives an error. 

        INPUTS
        ------
            ``args`` are all the arguments to save. 
        """
        hyper_frame = next((arg for arg in args if type(arg) == HyperFrame), None)

        for arg in args:
            match arg:
                case ConfigData():
                    self.saveConfig(arg, 'config.json')
                    self.saveDefaultMethodConfigs(arg)
                case InTrainEvals():
                    self.saveDuringTrainEvals(arg, 'in_train_evals.pkl')
                case Models():
                    self.saveModels(arg, 'models.pt')
                case OriginalFigure():
                    self.saveOriginalFigure(arg)
                case MetricFigures():
                    self.saveMetricFigures(arg)
                case Metrics():
                    self.saveMetrics(arg, 'metrics.pkl')
                case MeanMetrics():
                    self.saveMetrics(arg, 'mean_metrics.pkl')
                case ModelFigures():
                    try:
                        self.saveModelFigures(arg, hyper_frame) 
                    except:
                        print('Couldn\'t save model figures!')  
        assert hyper_frame is not None, '\'hyper_frame\' is None for saving unpruned stuff'

    
    def saveEverythingPrun(
        self, 
        method_config: PrunConfig | MethodPrunConfig, 
        *args
    ):
        """Saves every given argument at the dependencie of what class it is.
        If the hyperframe and method config is not in the arguments, the code gives an error. 
        It also gives an error, if method_config is not an instance of 'MethodPrunConfig' 
        or 'PrunConfig'.

        INPUTS
        ------
            ``method_config`` is the pruning method config. 

            ``args`` are all the arguments to save. 
        """
        method_str = get_method_str(method_config)

        self.directory = os.path.join(self.orig_directory, 'Pruning', method_str)
        createDir(self.directory)
        self.updateDirectorys(self.directory)
        self.saveMethodConfig(method_config, 'method_config.json')

        prun_hyperframe = next((arg for arg in args if type(arg) == HyperFrame), None)
        
        hyperframe_needed = False
        for arg in args:
            match arg:
                case Models():
                    self.saveModels(arg, save_name=f'models_{method_str}.pt') 
                case ModelsNotRemovedNodes():
                    self.saveModels(arg, save_name=f'nrn_models_{method_str}.pt') 
                case Metrics():
                    self.savePrunFrames(arg, method_str, 'metrics.pkl')
                case MeanMetrics():
                    self.savePrunFrames(arg, method_str, 'mean_metrics.pkl')
                case Sparsities():
                    self.savePrunFrames(arg, method_str, 'sparsities.pkl')
                case IterMetrics():
                    self.savePrunFrames(arg, method_str, 'iter_metrics.pkl')
                case IterMeanMetrics():
                    self.savePrunFrames(arg, method_str, 'iter_mean_metrics.pkl')
                case IterSparsity():
                    self.savePrunFrames(arg, method_str, 'iter_sparsity.pkl')
                case MetricFigures():
                    self.savePrunMetricsFiures(
                        arg, 
                        [f'mean_metrics_{method_str}'] + 
                        [f'{metric_name}_correlations_{method_str}' for metric_name in METRIC_NAMES]
                    )
                case Draws():
                    hyperframe_needed = True
                    try:
                        self.saveDraws(arg, prun_hyperframe, method_str)
                    except:
                        print('Couldn\'t save draws!')  
                case Graphs():
                    hyperframe_needed = True
                    try:
                        self.saveGraphs(arg, prun_hyperframe, method_str)
                    except:
                        print('Couldn\'t save Graphs!')  
                case IterPrunFigures():
                    hyperframe_needed = True
                    try:
                        self.saveSparsityIterFigures(arg, prun_hyperframe, method_str)
                    except:
                        print('Couldn\'t save iteration pruned figures!')  
        if hyperframe_needed:
            assert prun_hyperframe is not None, '\'hyper_frame\' is None for saving pruned stuff'
        

    def saveMultiNetPlotsAndIllus(
        self,
        *plots_and_illus_tuple
    ):
        """

        INPUTS
        ------
            ``method_config`` is the pruning method config. 

            ``args`` are all the arguments to save. 
        """
        for plots_and_illus in plots_and_illus_tuple:
            folder = next((arg for arg in plots_and_illus if type(arg) == str), None)
            self.directory = os.path.join(self.orig_directory, folder)
            self.updateDirectorys(self.directory)
            self.saveEverythingUnprun(*plots_and_illus)



    def saveMultiPrunPlotsAndIllus(
        self,
        *plots_and_illus_tuple
    ): 
        """

        INPUTS
        ------
            ``method_config`` is the pruning method config. 

            ``args`` are all the arguments to save. 
        """
        for plots_and_illus in plots_and_illus_tuple:
            method_config = next((arg for arg in plots_and_illus if isinstance(arg, PrunConfig)), None)
            self.saveEverythingPrun(method_config, *plots_and_illus)


    def saveEveryPerformance(
        self,
        *performance_tuples
    ):  
        """Saves the original and pruned performance frames.

        INPUTS
        ------
            ``orig_performance``, ``orig_mean_performance``, 
            ``prun_performances`` and ``prun_mean_performances`` 
            are all the performance frames that should be saved.
        """
        for performance_tuple in performance_tuples: 
            name_add = ''.join(filter(lambda x: type(x)==str, performance_tuple))
            for arg in performance_tuple:
                if type(arg) is Performances:
                    self.savePerformances(arg, f'{name_add}performances.pkl')
                if type(arg) is MeanPerformances:
                    self.savePerformances(arg, f'{name_add}mean_performances.pkl')



    #====================================================================================================
    # SAVE MODELS
    #====================================================================================================
    def saveModels(
        self, 
        models: Models, 
        save_name: str
    ):
        """Saves the models and it's features in the directory '3D_Saves' or '2D_Saves' 
        in the folder {time_folder} under the file 'models' as checkpoint 
        '{run_index}run_{network}net. 

        INPUTS
        ------
            ``models`` are multi models and it's features that are saved.

            ``save_name`` is the file name of the savings.
        """
        createDir(self.directory)
        file_path = os.path.join(self.directory, save_name)  
        model_dict = {}
        for network, model_list in enumerate(models):
            for run_idx, (features, model) in enumerate(model_list):
                name = f'{run_idx+1}run_{network}net'
                model_name = f'{name}_model'
                features_name = f'{name}_features'
                model_dict[model_name] = model.state_dict()
                model_dict[features_name] = features
        torch.save(model_dict, file_path)



    #====================================================================================================
    # SAVE FIGURES
    #====================================================================================================
    def saveOriginalFigure(
        self, 
        fig: OriginalFigure
    ):
        """Saves the original function figure in the directory 
        '3D_Saves' or '2D_Saves' in different folders and files.

        INPUTS
        ------
            ``fig_original`` is the figure of the original function.
        """
        # original function and metrics overview seperatly saved
        file_name = 'original_fun'
        path = os.path.join(self.directory, file_name)
        _save_figure(fig, path)


    def saveMetricFigures(
        self, 
        figures: MetricFigures
    ):
        """Saves the figures of the metrics in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/MetricsPlots' in the files 
        '{run_index}run_metrics.svg' or 'mean_metrics.svg'. 

        INPUTS
        ------
            ``figures`` are multi model metrics figures that are saved.
        """
        createDir(self.metrics_plots_folder)

        # save mean metrics figures
        file_name_mean = 'mean_metrics'
        path_mean = os.path.join(self.metrics_plots_folder, file_name_mean)
        _save_figure(figures[-1], path_mean)
        del figures[-1]

        # save run metrics figures
        for run_index, figure in enumerate(figures):
            file_name = f'{run_index+1}run_metrics'

            path = os.path.join(self.metrics_plots_folder, file_name)
            _save_figure(figure, path)


    def saveModelFigures(
        self, 
        figures: ModelFigures, 
        hyper_frame: HyperFrame
    ):
        """Saves the figures of the models in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/ModelPlots' in the file 
        '{run_index}run_{layer}layer_{nodes}nodes.svg'. 

        INPUTS
        ------
            ``figures`` are multi model figures that are saved.

            ``hyper_frame`` is the HyperFrame dataclass with frames of 
            multi networks of type DataFrame.
        """
        createDir(self.models_plots_folder)
        for network, figure_list in enumerate(figures):
            for run_idx, figure in enumerate(figure_list):
                layer = hyper_frame.iloc[network, 0]
                nodes = hyper_frame.iloc[network, 1]
                file_name = f'{layer}layer_{nodes}nodes_{run_idx+1}run'

                path = os.path.join(self.models_plots_folder, file_name)
                _save_figure(figure, path)
        
    
    def savePrunMetricsFiures(
        self, 
        figures: MetricFigures, 
        file_names: list[str]
    ):
        """Saves the figure of the pruned model metrics in the directory 
        '3D_Saves' or '2D_Saves' in the folder '{time_folder}/Pruning/{method_str}/MetricsPlots' 
        in the file '{prun_name}_mean_metrics.svg'. 

        INPUTS
        ------
            ``figures`` is a list of multiple pruned model metrics figures that are saved.

            ``prun_names`` is a list of the file name of the saved plot.
        """
        createDir(self.models_plots_folder)
        for figure, file_name in zip(figures, file_names):
            path = os.path.join(self.models_plots_folder, file_name)
            _save_figure(figure, path)

    
    def saveSparsityIterFigures(
        self, 
        figures: IterPrunFigures,
        hyper_frame: HyperFrame,
        method_str: str
    ):
        """Saves the figures of the sparse models during iteration 
        in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/Pruning/{method_str}/SparsityPlots' in the file 
        '{run_index}run_{layer}layer_{nodes}nodes.svg'. 

        INPUTS
        ------
            ``figures`` is a IterPrunFigures class with the figures of 
            the mean metrics and it's errors of every architecture. 

            ``hyper_frame`` is a HyperFrame dataclass with frames of 
            multi networks of type DataFrame.

            ``method_str`` is the name of the pruning method as a string.
        """
        createDir(self.sparsity_folder)
        for network, figure in enumerate(figures):
            layer = hyper_frame.iloc[network, 0]
            nodes = hyper_frame.iloc[network, 1]
            file_name = f'{layer}layer_{nodes}nodes_{method_str}'

            path = os.path.join(self.sparsity_folder, file_name)
            _save_figure(figure, path)


    def savePerformanceFigures(
        self,
        figures: PerformanceFigures,
        changed_hyperparam: pd.Index,
    ):
        """Saves the figures of the Performance 
        in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/PerformancePlots' in the file 
        '{changed_hyperparam}{changed_hyperparam_name}.svg'. (e.g. 8nodes.svg)

        INPUTS
        ------
            ``figures`` are the PerformanceFigures of different pruning methods and 
            and model architectures.  

            ``changed_hyperparam`` is a Series of the changed hyperparameters 
            (different architectures).
        """
        createDir(self.performance_plots_folder)
        for network, figure in enumerate(figures):
            file_name = f'{changed_hyperparam[network]}{changed_hyperparam.name}'

            path = os.path.join(self.performance_plots_folder, file_name)
            _save_figure(figure, path)
            
            
    def savePrunMetricComparisonFigure(
        self,
        figure: Figure,
        file_name: str,
    ):
        createDir(self.metrics_plots_folder)
        path = os.path.join(self.metrics_plots_folder, file_name)
        _save_figure(figure, path)
        


    #====================================================================================================
    # SAVE TIKZ
    #====================================================================================================
    def saveTikz(
            self,
            figure: Figure,
            dir: str, 
            file_name: str,
            legend_cols: int = 1,
    ):
        def _set_ncols(axes, legend_cols):
            if type(axes) == list:
                for ax in axes:
                    _set_ncols(ax, legend_cols)
            else:
                legend = axes.get_legend()
                if legend is not None:
                    legend._ncol = legend_cols


        path = os.path.join(dir, f'{file_name}.tikz')

        axes = figure.get_axes()
        _set_ncols(axes, legend_cols)

        width, height = figure.get_size_inches()
        tikz_code = tikzplt.get_tikz_code(figure, axis_width='\\textwidth', axis_height=f'{height/width}\\textwidth')
        
        # Save the TikZ code to a file
        with open(path, 'w') as f:
            f.write(tikz_code)



    #====================================================================================================
    # SAVE FRAMES
    #====================================================================================================
    def saveFrame(
        self, 
        frame: pd.DataFrame,
        file_name: str,
        method_config: PrunConfig | MethodPrunConfig = None,
        add_to_frame: bool = False
    ):
        """Saves the given frame as a pickle file in the given folder with the given file_name. 
        If a method_config is given then it adds the methods as indexes to the frame. 
        if add_to_frame is True, it adds the frame to an already exitsting saved frame, 
        with the same file_name in the same folder.


        INPUTS
        ------
            ``frame`` is the frame that should be saved. 

            ``folder`` is the folder where the frame should be saved. 

            ``file_name`` is the name of the file that should be saved. 

            ``method_config`` is a PruningMethod class for adding the method as indexes 
            to the frame.  

            ``add_to_frame`` is True if the frame should be added to an existing frame. 
        """
        createDir(self.frames_folder)
        file_path = os.path.join(self.frames_folder, file_name)

        if method_config is not None:
            frame = add_method_to_frame(frame, method_config=method_config)

        if not add_to_frame:
            if os.path.exists(file_path):
                os.remove(file_path)
            frame.to_pickle(file_path)
            if self.print_savings and os.path.exists(file_path):
                print(f'{file_name}: \n{frame}')
        else:
            add_to_existing_frame_file(
                frame, file_path, print_frame=self.print_savings
            )
            


    def saveMetrics(
        self, 
        metrics: Metrics | MeanMetrics,
        file_name: str
    ):
        """Saves the metrics frame as a pickle file in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/Metrics'.

        INPUTS
        ------
            ``metrics`` is a frame with the metrics for multiple architectures inside. 

            ``file_name`` is the name of the file that should be saved. 
        """
        self.saveFrame(
            frame=metrics, 
            file_name=file_name
        )


    def saveDuringTrainEvals(
        self, 
        during_train_evals: InTrainEvals,
        file_name: str
    ):
        """Saves the during training evals frame as a pickle file in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/InTrainEvals'.

        INPUTS
        ------
            ``during_train_evals`` is a frame with the evals made during training.  

            ``file_name`` is the name of the file that should be saved. 
        """
        self.saveFrame(
            frame=during_train_evals, 
            file_name=file_name
        )


    def savePerformances(
        self, 
        performances: MeanPerformances,
        file_name: str
    ):
        """Saves the performances frame as a pickle file in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/Performances'.

        INPUTS
        ------
            ``performances`` is a frame with the performances for every pruning method used inside. 

            ``file_name`` is the name of the file that should be saved. 
        """
        self.saveFrame(
            frame=performances, 
            file_name=file_name,
            add_to_frame=True,
        )
    

    def savePrunFrames(
        self, 
        frame: pd.DataFrame,
        method_str: str,
        file_name: str
    ):
        """Saves the pruning metrics frame as a pickle file in the directory '3D_Saves' or '2D_Saves' 
        in the folder '{time_folder}/Pruning/{method_str}/Metrics'.

        INPUTS
        ------
            ``frame`` is a frame with the metrics for multiple architectures inside. 

            ``method_str`` is the pruning method string to add the method as indexes to the frame. 

            ``file_name`` is the name of the file that should be saved. 
        """
        frame.Name = method_str
        self.saveFrame(
            frame=frame, 
            file_name=file_name,  
            add_to_frame=False
        )



    #====================================================================================================
    # SAVE LIST
    #====================================================================================================
    def saveList(
        self, 
        data_list: list,
        file_name: str
    ): 
        """Saves the given list as a pickle file as the given file_name in the directory 
        '3D_Saves' or '2D_Saves' in the folder '{time_folder}/Pruning/{method_str}'.

        INPUTS
        ------
            ``data_list`` is a frame with the performances for multiple architectures inside. 

            ``file_name`` is the name of the file that should be saved. 
        """
        file_path = os.path.join(self.directory, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(data_list, file)



    #====================================================================================================
    # SAVE ILLUSTRATIONS
    #====================================================================================================
    def saveGraphs(
        self, 
        graphs: Graphs, 
        hyper_frame: HyperFrame, 
        method_str: str,
        show: bool = False
    ):
        """Saves the graphs as svgs in the file 
        '{layer}layer_{nodes}nodes_{run_idx+1}run_{method_str}' 
        in the directory '3D_Saves' or '2D_Saves' in the folder 
        '{time_folder}/Pruning/{method_str}/Graphs'.

        INPUTS
        ------
            ``graphs`` are the Graphs of every architecture of a specific pruning method. 

            ``hyper_frame`` is the hyper frame with all hyperparameters inside. 

            ``method_str`` is the pruning method as a string.  

            ``show`` is True if every Graph should be shown. 
        """
        createDir(self.graphs_folder)
        for network, graph_runs in enumerate(graphs):
            for run_idx, dot in enumerate(graph_runs):
                layer = hyper_frame.iloc[network, 0]
                nodes = hyper_frame.iloc[network, 1]
                dot.format = 'svg'
                path = os.path.join(self.graphs_folder, f'{layer}layer_{nodes}nodes_{run_idx+1}run_{method_str}')
                dot.render(path, view = show) 

    
    def saveDraws(
        self, 
        draws: Draws, 
        hyper_frame: HyperFrame,
        method_str: str
    ):
        """Saves the draws as svgs in the file 
        '{layer}layer_{nodes}nodes_{run_idx+1}run_{method_str}' 
        in the directory '3D_Saves' or '2D_Saves' in the folder 
        '{time_folder}/Pruning/{method_str}/Draws'.

        INPUTS
        ------
            ``draws`` are the matrix illustrations of every architecture of a specific pruning method. 

            ``hyper_frame`` is the hyper frame with all hyperparameters inside. 

            ``method_str`` is the pruning method as a string.
        """
        createDir(self.draws_folder)
        for network, draws_run in enumerate(draws):
            for run_idx, draw in enumerate(draws_run):
                layer = hyper_frame.iloc[network, 0]
                nodes = hyper_frame.iloc[network, 1]
                path = os.path.join(self.draws_folder, f'{layer}layer_{nodes}nodes_{run_idx+1}run_{method_str}')
                draw.writePDFfile(path)

            

    #====================================================================================================
    # SAVE JSON
    #====================================================================================================
    def saveConfig(
        self, 
        config: ConfigData, 
        file_name: str
    ):
        """Saves hyperparameter as a JSON file in the directory '3D_Saves' or 
        '2D_Saves' in the folder {time_folder}

        INPUTS
        ------
            ``config`` is the configuration file of the models of type ConfigData.

            ``config_str`` is the name of the JSON file that is generated. (e.g. config.json)

        RETURN
        ------
            ``file_exists`` is True if the file is saved and exists. 
        """
        multiNets = {
            'networks': config.networks,
            'runs': config.runs,
            'loader_reprod': config.loader_reprod,
            'make_plots': config.make_plots
        }
        data = {
            'train_size': config.train_size,
            'test_size': config.test_size
        }
        neuralNet = {
            'degree': config.degree,
            'inputs': config.inputs,
            'outputs': config.outputs,
            'layer': config.layer,
            'nodes': config.nodes
        }
        training = {
            'learning_rate': config.learning_rate,
            'epochs': config.epochs,
            'batch_size': config.batch_size
        }

        multiNetSteps = {
            'layer_step': config.layer_step,
            'node_step': config.node_step
        }
        noise = {
            'mean': config.mean,
            'std': config.std
        }
        config_dict = {
            'multiNets': multiNets,
            'data': data,
            'neuralNet': neuralNet,
            'training': training,
            'multiNetStep': multiNetSteps,
            'noise': noise
        }

        file_path = os.path.join(self.directory, file_name)
        file_exists = save_JSON(config_dict, file_path)
        return file_exists


    def saveMethodConfig(
        self, 
        method_config: PrunConfig | MethodPrunConfig,
        file_name: str
    ):
        """Saves the method config json in the default dictionary '3D_Saves' or 
        '2D_Saves' in the folder {time_folder} with the {file_name}. 

        INPUTS
        ------
            ``method_config`` is the configuration file of the pruning method. 

            ``file_name`` is the name of the file, ending included. 

        RETURN
        ------
            ``file_exists`` is True if the file is saved and exists. 
        """
        config_dict = {
            'type': method_config.type,
            'networks': method_config.networks,
            'layer': method_config.layer,
            'nodes': method_config.nodes,
            'amount': method_config.amount,
            'bias_or_weight': method_config.bias_or_weight,
            'remove_nodes': method_config.remove_nodes,
            'prun_prev_params': method_config.prun_prev_params,
            'prun_next_params': method_config.prun_next_params
        }

        if type(method_config) is MethodPrunConfig:
            config_dict['iterations'] = method_config.iterations
            config_dict['schedule'] = method_config.schedule
            config_dict['last_iter_npp'] = method_config.last_iter_npp
            config_dict['method'] = method_config.method

        file_path = os.path.join(self.directory, file_name)
        file_exists = save_JSON(config_dict, file_path)
        return file_exists


    def saveAllMethodConfig(
        self,
        config: ConfigData,
        all_method_config: AllMethodPrunConfig | AllPrunConfig,
        file_name: str,

    ):  
        if type(all_method_config) is AllPrunConfig:
            method_config = PrunConfig(
                all_method_config.type, 
                config.networks, 
                config.layer,
                config.nodes,
                all_method_config.amount,
                all_method_config.bias_or_weight,
                all_method_config.remove_nodes,
                all_method_config.prun_prev_params,
                all_method_config.prun_next_params,
            )
        if type(all_method_config) is AllMethodPrunConfig:
            method_config = MethodPrunConfig(
                all_method_config.type, 
                config.networks, 
                config.layer,
                config.nodes,
                all_method_config.amount,
                all_method_config.bias_or_weight,
                all_method_config.remove_nodes,
                all_method_config.prun_prev_params,
                all_method_config.prun_next_params,
                all_method_config.iterations,
                all_method_config.schedule,
                all_method_config.last_iter_npp,
                all_method_config.method,
            )
        self.saveMethodConfig(method_config, file_name)


    def saveDefaultMethodConfigs(
        self,
        config: ConfigData
    ):
        all_method_config = AllMethodPrunConfig(
            'l1_unstr', 
            0.2,
            'weight',
            True,
            True,
            False,
            0,
            'binom',
            False,
            'finetune',
        )
        self.saveAllMethodConfig(config, all_method_config, 'prun_method_config.json')

        all_method_config = AllPrunConfig(
            'l1_unstr', 
            0.2,
            'weight',
            True,
            True,
            False,
        )
        self.saveAllMethodConfig(config, all_method_config, 'prun_type_config.json')



def save_JSON(config_dict: dict, file_path: str):
    with open(file_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=3, sort_keys=False)
    return os.path.exists(file_path)



def add_method_to_frame(frame: pd.DataFrame, method_config: PrunConfig | MethodPrunConfig):
    amount = method_config.amount
    bias_or_weight = method_config.bias_or_weight
    model_type = method_config.type

    node_prun = ''
    if method_config.remove_nodes:
        node_prun += 'rn_'
    if method_config.prun_prev_params:
        node_prun += 'pp_'
    if method_config.prun_next_params:
        node_prun += 'np_'
    if not node_prun:
        node_prun = '_'

    if type(method_config) is MethodPrunConfig:
        if method_config.last_iter_npp:
            node_prun += 'lnp_'
        iterations = method_config.iterations
        schedule = method_config.schedule if method_config.schedule!='' else 'None'
        method = method_config.method

    if type(method_config) is PrunConfig:
        iterations = 0
        schedule = 'None'
        method = 'magnitude'

    frame.index = add_Index_to_MultiIndex(frame.index, (amount,), 'amount')
    frame.index = add_Index_to_MultiIndex(frame.index, (bias_or_weight,), 'structure')
    frame.index = add_Index_to_MultiIndex(frame.index, (node_prun,), 'node_prun')
    frame.index = add_Index_to_MultiIndex(frame.index, (schedule,), 'schedule')
    frame.index = add_Index_to_MultiIndex(frame.index, (iterations,), 'iterations')
    frame.index = add_Index_to_MultiIndex(frame.index, (model_type,), 'type')
    frame.index = add_Index_to_MultiIndex(frame.index, (method,), 'method') 
    return frame



def save_default_configs():
    savings = Saves(save_directory=os.path.dirname(__file__))
    savings.saveConfig(get_default_config(2), 'hyperparams2d.json')
    savings.saveConfig(get_default_config(3), 'hyperparams3d.json')
    


def get_default_config(dim: int):
    if dim == 2:
        return ConfigData(
            1, 1, False, False, 50, 400, 1, 1, 1, 2, 4, 0.01, 5000, 16, 0, 1, 0.0, 4.0
        )
    elif dim == 3:
        return ConfigData(
            1, 1, False, False, 14, 100, 1, 2, 1, 2, 4, 0.01, 5000, 32, 0, 1, 0.0, 3.0
        )
    else:
        raise(ValueError('Dimension not 2 or 3!', dim))

    
    
def get_method_str(method_config: PrunConfig | MethodPrunConfig):
    node_rem_add = '_rn' if method_config.remove_nodes else ''
    prev_param_add = '_pp' if method_config.prun_prev_params else ''
    next_param_add = '_np' if method_config.prun_next_params else ''       

    middle_method_str = f'{method_config.type}{node_rem_add}{prev_param_add}{next_param_add}'
    end_method_str = f'{method_config.bias_or_weight}_{method_config.amount}'.replace('.', '_')
    if type(method_config) is MethodPrunConfig:
        last_iter_npp = '_lnp' if method_config.last_iter_npp else ''
        schedule_add = f'_{method_config.schedule}' if method_config.schedule else ''
        return f'{method_config.method}{schedule_add}_{method_config.iterations}_{middle_method_str}{last_iter_npp}_{end_method_str}'
    elif type(method_config) is PrunConfig:
        return f'magnitude_0_{middle_method_str}_{end_method_str}'
    else:
        raise(ValueError('\'method_str\' not an instance of \'MethodPrunConfig\' or \'PrunConfig\''))




def _save_figure(figure: Figure, path):
    # figure.savefig(path + '.pgf', format='pgf')
    figure.savefig(path + '.pdf', format='pdf')