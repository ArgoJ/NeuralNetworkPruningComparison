import sys
import os
import pandas as pd


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from mainEvals import makeModelEvals
from Loads import Loadings, loadConfig
from Helpers import selectDevice
from CustomExceptions import DimensionException
from DataClasses import Models

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from PrunHelpers import getPrunConfig, removeNodes





def mainPrunEvals(base_path: str, prun_folder: str):
    loaded_config = loadConfig(os.path.join(base_path, 'config.json'))
    load_cls = Loadings(base_path)
    load_cls.setPrunMethodDirectorys(prun_folder)
    
    method_config = load_cls.loadMethodConfig()
    model_config = getPrunConfig(loaded_config, method_config)
    
    # chooses if 2d or 3d
    if loaded_config.inputs==1:
        # 2D stuff
        data_cls = Data2d(loaded_config)
    elif loaded_config.inputs==2:
        # 3D stuff
        data_cls = Data3d(loaded_config)
    else:
        raise(DimensionException(loaded_config.inputs + 1))
    
    models = load_cls.loadAllModels_ForPrun(model_config, loaded_config, use_notRemNode_models=True)
    metrics = makeModelEvals(data_cls, model_config, models)
    
    new_models = Models([] for _ in models)
    for net_idx, models_net in enumerate(models):
        for run_idx, (features, model) in enumerate(models_net):
            if net_idx == 10 and (run_idx in [2, 9, 11, 13]):
                pass
            new_models[net_idx].append(removeNodes(features, model)) 
    # new_models = Models([removeNodes(features, model) for features, model in models_net] for models_net in models)
    new_metrics = makeModelEvals(data_cls, model_config, new_models)
    
    with pd.option_context('display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 3,
        ):
        level = 'old_nodes'
        level2 = 'nodes'
        key = 66
        print(f'Prun Metrics:\n{metrics.xs(key=key, axis=0, level=level2)}\n\n\n')
        print(f'Prun Metrics:\n{new_metrics.xs(key=key, axis=0, level=level2)}\n\n\n')
        
    return metrics, new_metrics






if __name__ == "__main__":
    selectDevice()
    prun_path = input('type in the path\n').replace('"', '')
    base_path = os.path.dirname(os.path.dirname(prun_path)).replace('_Saves_prun1', '_Saves').replace('_Saves_prun2', '_Saves')
    prun_folder = os.path.basename(prun_path)
    mainPrunEvals(base_path, prun_folder)