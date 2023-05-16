import json
import os
import torch
import numpy as np
import pandas as pd

from copy import deepcopy


from CustomExceptions import DimensionException, NotCorrectlyLoaded
from DataClassesJSON import *
from DataClasses import Models, ModelsNotRemovedNodes
from Constants import *
from NeuralNetwork import Model
from Helpers import find_model, comparissonIndexes



CURR_DIR = os.path.dirname(__file__)


class Loadings:
    def __init__(
        self, 
        directory: str
    ) -> None:
        """Initialises a ModelLoads class with the directory created out of dimensions and 
        time_folder. Also loads the configs of the models.
        
        INPUTS
        ------
            ``directory`` is the path where the loaded files are.
        """
        self.directory = directory
        self.orig_directory = deepcopy(self.directory)

        self.updateDirectorys(self.directory)
        self.prun_method_directory = None


    def updateDirectorys(
        self,
        directory: str,
    ):
        self.directory = directory
        self.prun_directory = os.path.join(self.directory, 'Pruning')
        self.frames_directory = os.path.join(self.directory, 'Frames')

        
    def setPrunMethodDirectorys(
        self, 
        folder: str,
    ):
        self.prun_method_directory = os.path.join(self.prun_directory, folder)
        self.prun_frames_directory = os.path.join(self.prun_method_directory, 'Frames')

        


    #====================================================================================================
    # MODELS
    #====================================================================================================
    def loadModel(
        self, 
        run_idx: int, 
        network: int,
        models_path: str
    ) -> tuple[list[int], Model]:
        """Inizialise a model with same configs as the the model to load
        and then load the parameters in the model. 
        Also loads the features in the checkpoint '{name}_features'

        INPUTS
        ------
            ``run_index``, ``network`` are in the name of the plots
            and specify the model that's loaded.

            ``models_path`` is the path where the model is saved. 

        RETURN
        ------
            ``features`` are the features as a list of ints of the loaded model.

            ``model`` with inserted loaded parameters of type Model.
        """
        checkpoint = torch.load(models_path)
        name = f'{run_idx+1}run_{network}net'
    
        features = checkpoint[f'{name}_features']
        model = Model(features)

        model.load_state_dict(
            checkpoint[f'{name}_model'])
        return features, model


    def loadOriginalModel(
        self, 
        run_index: int, 
        network: int
    ) -> tuple[list[int], Model]:
        """Inizialise a model with same configs as the the model to load
        and then load the parameters in the model.

        INPUTS
        ------
            ``run_index``, ``network`` are in the name of the plots
            and specify the model that's loaded.

        RETURN
        ------
            ``model`` with inserted parameters of type Model  
        """
        return self.loadModel(
            run_index, 
            network,
            models_path = os.path.join(self.orig_directory, 'models.pt')
        )
    

    def loadAllModels(
        self,
        config: ConfigData,
        rand_run_idx = None
    ):  
        models = Models([] for _ in range(config.networks))
        model_path, _ = find_model(self.directory)
        for network in range(config.networks):  
            for run_index in (range(config.runs) if rand_run_idx is None else (rand_run_idx,)):
                models[network].append(
                    self.loadModel(
                        run_index, 
                        network,
                        model_path
                    )
                )
        return models

    
    def loadAllModels_ForPrun(
        self,
        model_config: ConfigData,
        loaded_config: ConfigData,
        use_notRemNode_models = False,
        rand_run_idx = None
    ) -> Models | ModelsNotRemovedNodes: 
        paths = find_model(self.prun_method_directory)
        if use_notRemNode_models:
            models_inst = ModelsNotRemovedNodes 
            models_path = paths[1]
        else: 
            models_inst = Models
            models_path = paths[0]

        models = models_inst([] for _ in range(model_config.networks))
        network_idxs = comparissonIndexes(loaded_config, model_config)
        for network in range(model_config.networks):  
            for run_index in (range(model_config.runs) if rand_run_idx is None else (rand_run_idx,)):
                models[network].append(
                    self.loadModel(
                        run_index, 
                        network_idxs[network],
                        models_path
                    )
                )
        return models



    #====================================================================================================
    # FRAMES
    #====================================================================================================
    def loadFrame(
        self, 
        path: str,
        file_name: str
    ) -> pd.DataFrame:
        """Get the frame of the given path and filename. 

        INPUTS
        ------
            ``path`` is the path where the file is.
            
            ``file_name`` is the file name of the .

        RETURN
        ------
            ``frame`` is the read DataFrame.
        """
        file_path = os.path.join(path, file_name)
        return pd.read_pickle(file_path)


    def loadMetrics(self):
        """

        RETURN
        ------
            ``metrics``
            
            ``mean_metrics`` 
        """
        metrics = self.loadFrame(self.frames_directory, 'metrics.pkl')
        mean_metrics = self.loadFrame(self.frames_directory, 'mean_metrics.pkl')
        return metrics, mean_metrics


    def loadInTrainEvals(self):
        """

        RETURN
        ------
            ``in_train_evals``
        """
        return self.loadFrame(self.frames_directory, 'in_train_evals.pkl')

    
    def loadPrunMetrics(self):
        """

        RETURN
        ------
            ``prun_metrics`` 
            
            ``prun_mean_metrics`` 
        """
        prun_metrics = self.loadFrame(self.prun_frames_directory, 'metrics.pkl')
        prun_mean_metrics = self.loadFrame(self.prun_frames_directory, 'mean_metrics.pkl')
        return prun_metrics, prun_mean_metrics
    
    
    def loadPrunSparsities(self):
        """

        RETURN
        ------
            ``sparsities``
        """
        return self.loadFrame(self.prun_frames_directory, 'sparsities.pkl')

    
    def loadIterMetrics(self):
        """

        RETURN
        ------
            ``prun_metrics`` 
            
            ``prun_mean_metrics`` 
        """
        iter_metrics = self.loadFrame(self.prun_frames_directory, 'iter_metrics.pkl')
        iter_mean_metrics = self.loadFrame(self.prun_frames_directory, 'iter_mean_metrics.pkl')
        return iter_metrics, iter_mean_metrics


    def loadIterSparsity(self):
        """

        RETURN
        ------
            ``iter_sparsity`` 
        """
        iter_sparsity = self.loadFrame(self.prun_frames_directory, 'iter_sparsity.pkl')
        return iter_sparsity


    def loadPerformances(
        self, 
        file_name_frontadd: str = '',
    ):
        """Gets the saved performance frames. 

        INPUTS
        ------
            ``file_name_frontadd`` is added infront of the file name. 
            Can be 'old_' or just NOTHING.

        RETURN
        ------
            ``orig_performances``, ``orig_mean_performances`` are the performances and 
            the averages of the original networks. 
        """
        orig_performances = self.loadFrame(self.frames_directory, file_name_frontadd + 'performances.pkl')
        orig__mean_performances = self.loadFrame(self.frames_directory, file_name_frontadd + 'mean_performances.pkl')
        return orig_performances, orig__mean_performances
    
    
    def loadPrunPerformances(
        self, 
        file_name_frontadd: str = '',
    ):
        """Gets the saved performance frames. 

        INPUTS
        ------
            ``file_name_frontadd`` is added infront of the file name. 
            Can be 'old_' or just NOTHING.

        RETURN
        ------
            ``prun_performances``, ``prun_mean_performances`` are the performances and 
            the averages of the pruned networks. 
        """
        prun_performances = self.loadFrame(self.frames_directory, file_name_frontadd + 'pruned_performances.pkl')
        prun_mean_performances = self.loadFrame(self.frames_directory, file_name_frontadd + 'pruned_mean_performances.pkl')
        return prun_performances, prun_mean_performances


    #====================================================================================================
    # CONFIGS
    #====================================================================================================
    def loadBaseConfig(self) -> ConfigData:
        file_path = os.path.join(self.directory, 'config.json')
        return loadConfig(file_path)

    
    def loadBasePrunMethodConfig(self) -> PrunConfig | MethodPrunConfig:
        return getPrunMethod(self.directory)

    
    def loadMethodConfig(self) -> PrunConfig | MethodPrunConfig:
        """Gets the pruning method config of the already pruned models in the given folder. 

        INPUTS
        ------
            ``folder`` is the pruning folder where the method_config is saved.

        RETURN
        ------
            ``method_config`` is an instance of PrunConfig where the pruning method is configurated.
        """
        file_path = os.path.join(self.prun_method_directory, 'method_config.json')
        if not os.path.exists(file_path):
            raise(ValueError('method_config doesn\'t exist!'))
        try:
            method_config = loadPrunConfig(file_path)
        except:
            pass
        try:
            method_config = loadMethodPrunConfig(file_path)
        except:
            pass
        if 'method_config' not in locals():
            raise(ValueError('method_config is not a MethodPrunConfig or PrunConfig!'))
        return method_config


    #====================================================================================================
    # DIRECTORY GETTER
    #====================================================================================================
    def getOrigBaseDirectory(self) -> str:
        return self.orig_directory
    
    def getBaseDirectory(self) -> str:
        return self.directory

    def getFramesDirectory(self) -> str:
        return self.frames_directory

    def getPrunDirectory(self) -> str:
        return self.prun_directory

    def getPrunMethodDirectory(self) -> str:
        return self.prun_method_directory



def loadConfigWithDim(dimension: int) -> ConfigData:
    """Load hyperparameters as config data of type ConfigData. 
    Raise LoadJSONException, if the dimension are not correct.

    INPUTS
    ------
        ``dimension`` is the dim of the hyperparam file, that is used. 

    RETURN
    ------
        ``config`` is the configuration file of the models of type ConfigData.
    """
    if 3<dimension or dimension<2: 
        raise DimensionException(dimension)
    filename = f'hyperparams{dimension}d.json'
    path = os.path.join(CURR_DIR, filename)
    return loadConfig(path)



def loadConfig(file_path: str) -> ConfigData:
    """Loads the config.json as a dictionary
    and put it in ConfigData.

    INPUTS
    ------
        ``file_path`` is the path where the config file is.

    RETURN
    ------
        ``config`` is the configuration file of the models of type ConfigData.
    """
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
        
    return ConfigData(
        json_dict['multiNets']['networks'],
        json_dict['multiNets']['runs'],
        json_dict['multiNets']['loader_reprod'],
        json_dict['multiNets']['make_plots'],
        json_dict['data']['train_size'],
        json_dict['data']['test_size'],
        json_dict['neuralNet']['degree'],
        json_dict['neuralNet']['inputs'],
        json_dict['neuralNet']['outputs'],
        json_dict['neuralNet']['layer'],
        json_dict['neuralNet']['nodes'],
        json_dict['training']['learning_rate'],
        json_dict['training']['epochs'],
        json_dict['training']['batch_size'],
        json_dict['multiNetStep']['layer_step'],
        json_dict['multiNetStep']['node_step'],
        json_dict['noise']['mean'],
        json_dict['noise']['std']
    )


def loadBasicDimConfig(file_path: str) -> BaseConfig:
    """Loads the config.json as a dictionary
    and put it in ConfigData.

    INPUTS
    ------
        ``file_path`` is the path where the config file is.

    RETURN
    ------
        ``basic_dim_config`` is the configuration file of the models of type ConfigData.
    """
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
        
    return BaseConfig(
        json_dict['multiNets']['networks'],
        json_dict['multiNets']['runs'],
        json_dict['multiNets']['loader_reprod'],
        json_dict['multiNets']['make_plots'],
        json_dict['data']['train_size'],
        json_dict['data']['test_size'],
        json_dict['neuralNet']['inputs'],
        json_dict['neuralNet']['outputs'],
        json_dict['training']['learning_rate'],
        json_dict['training']['epochs'],
        json_dict['training']['batch_size'],
        json_dict['noise']['mean'],
        json_dict['noise']['std']
    )


def loadArchitectureConfig(file_path: str) -> ArchitectureConfig:
    """Loads the config.json as a dictionary
    and put it in ConfigData.

    INPUTS
    ------
        ``file_path`` is the path where the config file is.

    RETURN
    ------
        ``basic_dim_config`` is the configuration file of the models of type ConfigData.
    """
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
        
    return ArchitectureConfig(
        json_dict['neuralNet']['degree'],
        json_dict['neuralNet']['layer'],
        json_dict['neuralNet']['nodes'],
        json_dict['multiNetStep']['layer_step'],
        json_dict['multiNetStep']['node_step']
    )

def makeConfigDataOfOthers(basic_dim_config: BaseConfig, architecture_config: ArchitectureConfig):
    """Makes a ConfigData out of BasicDimConfig and ArchitectureConfig
    """
    return ConfigData(
        basic_dim_config.networks,
        basic_dim_config.runs,
        basic_dim_config.loader_reprod,
        basic_dim_config.make_plots,
        basic_dim_config.train_size,
        basic_dim_config.test_size, 
        architecture_config.degree,
        basic_dim_config.inputs,
        basic_dim_config.outputs,
        architecture_config.layer,
        architecture_config.nodes,
        basic_dim_config.learning_rate,
        basic_dim_config.epochs,
        basic_dim_config.batch_size,
        architecture_config.layer_step,
        architecture_config.node_step,
        basic_dim_config.mean,
        basic_dim_config.std
    )



def loadPrunConfig(file_path: str) -> PrunConfig:
    """Loads the config.json as a dictionary
    and put it in ConfigData.

    INPUTS
    ------
        ``file_path`` is the path where the config file is.

    RETURN
    ------
        ``method_config`` is the configuration file of the models of type ConfigData.
    """
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
    
    return PrunConfig(
        json_dict['type'],
        json_dict['networks'],
        json_dict['layer'],
        json_dict['nodes'],
        json_dict['amount'],
        json_dict['bias_or_weight'],
        json_dict['remove_nodes'],
        json_dict['prun_prev_params'],
        json_dict['prun_next_params'],
    )



def loadMethodPrunConfig(file_path: str) -> MethodPrunConfig:
    """Loads the config.json as a dictionary
    and put it in ConfigData.

    INPUTS
    ------
        ``file_path`` is the path where the config file is.

    RETURN
    ------
        ``method_config`` is the configuration file of the models of type ConfigData.
    """
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
    
    return MethodPrunConfig(
        json_dict['type'],
        json_dict['networks'],
        json_dict['layer'],
        json_dict['nodes'],
        json_dict['amount'],
        json_dict['bias_or_weight'],
        json_dict['remove_nodes'],
        json_dict['prun_prev_params'],
        json_dict['prun_next_params'],
        json_dict['iterations'],
        json_dict['schedule'],
        json_dict['last_iter_npp'],
        json_dict['method'],
    )



def loadBaseMethodPrunConfig(config: ConfigData, file_path: str):
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
    
    return MethodPrunConfig(
        json_dict['type'],
        config.networks,
        config.layer,
        config.nodes,
        json_dict['amount'],
        json_dict['bias_or_weight'],
        json_dict['remove_nodes'],
        json_dict['prun_prev_params'],
        json_dict['prun_next_params'],
        json_dict['iterations'],
        json_dict['schedule'],
        json_dict['last_iter_npp'],
        json_dict['method'],
    )



def loadBasePrunConfig(config: ConfigData, file_path: str):
    with open(file_path) as json_file:
        json_dict = json.load(json_file)
    
    return PrunConfig(
        json_dict['type'],
        config.networks,
        config.layer,
        config.nodes,
        json_dict['amount'],
        json_dict['bias_or_weight'],
        json_dict['remove_nodes'],
        json_dict['prun_prev_params'],
        json_dict['prun_next_params']
    )



def loadBaseMethodConfig(config: ConfigData, file_path: str):
    try:
        method_config = loadBasePrunConfig(config, file_path)
    except:
        pass
    try:
        method_config = loadBaseMethodPrunConfig(config, file_path)
    except:
        pass
    if 'method_config' not in locals():
        raise(ValueError('method_config is not a MethodPrunConfig or PrunConfig!'))
    return method_config




def getDimensionSaveDirectory(
    dimension: int
): 
    """Checks if the dimension exists and get the dimension directory as 
    '*/3DSaves'. 

    INPUTS
    ------
        ``dimension`` is the dimension of the models. 

    RETURN
    ------
        ``directory`` is the dimenstoin directory. 
    """
    if 3<dimension or dimension<2: 
        raise DimensionException(dimension)
    parent_dir = os.path.dirname(CURR_DIR)
    return os.path.join(parent_dir, f'{dimension}D_Saves')



def getSavesDirectory(
    dimension: int, 
    time_folder: str
) -> os.path:
    """Checks if the dimension exists and get the directory of 
    the models. 

    INPUTS
    ------
        ``dimension`` is the dimension of the saved models. 

        ``time_folder`` is the folder of the saved models of type '%m_%d_%H_%M'. 

    RETURN
    ------
        ``directory`` of the models and config file.
    """
    return os.path.join(getDimensionSaveDirectory(dimension), time_folder)



def getPrunMethod(save_directory: str) -> PrunConfig | MethodPrunConfig | None:
    prun_method_name = input(
        '''Type 
        \'m\' for pruning with a method or 
        \'t\' for pruning with a type like magnitude pruning!
        '''
    )
    if 'm' in prun_method_name:
        return loadMethodPrunConfig(os.path.join(save_directory, 'prun_method_config.json'))
    elif 't' in prun_method_name:
        return loadPrunConfig(os.path.join(save_directory, 'prun_type_config.json'))
    else:
        print('no correct input')



def checkAllMetricsLoaded(
    loaded_config: ConfigData,
    prun_config: ConfigData,
    loaded_metrics: pd.DataFrame, 
    prun_metrics: pd.DataFrame
    ) -> bool:
    """

    INPUTS
    ------
        ``loaded_config`` is the configuration of the LOADED models 

        ``prun_config`` is the configuration of the PRUNED models. 

        ``loaded_metrics`` are the LOADED train and test metrics of every model. 

        ``loaded_metrics`` are the PRUNED train and test metrics of every model 

    RETURN
    ------
        ``all_loaded`` 
    """
    if loaded_config.runs != len(
        loaded_metrics.index.get_level_values('run').unique()
    ) or prun_config.runs != len(
        prun_metrics.index.get_level_values('run').unique()
    ):
        raise NotCorrectlyLoaded('RUNS')
    feature_name = loaded_metrics.index.get_level_values(0).name
    if loaded_config.networks != len(
        loaded_metrics.index.get_level_values(0).unique()
    ) or prun_config.networks != len(
        prun_metrics.index.get_level_values(f'old_{feature_name}').unique()
    ):
        raise NotCorrectlyLoaded('NETWORKS')
    print('Everything is correctly loaded!')
    return True
