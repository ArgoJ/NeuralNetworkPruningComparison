import os
import numpy as np


from Saves import Saves
from Loads import loadConfig, Loadings
from basicFrame import getMetricFrame
from Helpers import (
    selectDevice,
    calcHyperparams, 
    get_total_nodes,
)
from DataClassesJSON import ConfigData
from DataClasses import Models
from CustomExceptions import DimensionException
from NeuralNetwork import evalModel

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d



def makeModelEvals(data_cls: Data2d | Data3d, config: ConfigData, models: Models):
    hyper_frame = calcHyperparams(config)
    total_nodes = get_total_nodes(hyper_frame)
    metrics = getMetricFrame(
        total_nodes,
        config.runs
    )
    
    train_data = data_cls.getTrainData()
    test_data = data_cls.getTestData()
    
    for net_idx, models_net in enumerate(models):
        for run_idx, (features, model) in enumerate(models_net):
            metrics.iloc[run_idx + net_idx*config.runs] = np.append(evalModel(model, train_data), evalModel(model, test_data))
    
    return metrics
    




def mainEvals(base_path: str):
    config = loadConfig(os.path.join(base_path, 'config.json'))
    load_cls = Loadings(base_path)
    
    # chooses if 2d or 3d
    if config.inputs==1:
        # 2D stuff
        data_cls = Data2d(config)
    elif config.inputs==2:
        # 3D stuff
        data_cls = Data3d(config)
    else:
        raise(DimensionException(config.inputs + 1))
    models = load_cls.loadAllModels(config)
    metrics = makeModelEvals(data_cls, config, models)

    
        
    
    
    


if __name__ == "__main__":
    selectDevice()
    input_path = input('type in the path\n').replace('"', '')
    mainEvals(input_path)