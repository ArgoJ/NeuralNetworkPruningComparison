import pandas as pd
import numpy as np
import os, sys
from pathlib import Path


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from changeStuff import on_every_Prune_config_do
from Saves import Saves, get_method_str
from Loads import Loadings, loadConfig
from basicFrame import getSparsitiesFrame
from Helpers import calcHyperparams, get_total_nodes, get_new_total_nodes, add_new_index_chp

from Pruning.PrunHelpers import getPrunConfig, getSparsity




def makeModelSparsities(base_path: Path, prune_path: Path):
    loaded_config = loadConfig(os.path.join(base_path, 'config.json'))
    
    load_cls = Loadings(base_path)
    prun_base_path = os.path.dirname(os.path.dirname(prune_path))
    load_cls.updateDirectorys(prun_base_path)
    prun_folder = os.path.basename(prune_path)
    load_cls.setPrunMethodDirectorys(prun_folder)
    
    method_config = load_cls.loadMethodConfig()
    model_config = getPrunConfig(loaded_config, method_config)
    
    models = load_cls.loadAllModels_ForPrun(model_config, loaded_config, use_notRemNode_models=False)
    nrn_models = load_cls.loadAllModels_ForPrun(model_config, loaded_config, use_notRemNode_models=True)
    
    hyper_frame = calcHyperparams(model_config)
    total_nodes = get_total_nodes(hyper_frame)

    bias_or_weight = 'weight'
    
    sparsities = np.zeros((model_config.networks, model_config.runs))
    new_total_nodes = np.zeros((model_config.networks, model_config.runs))
    total_sparsities = np.zeros((model_config.networks, model_config.runs))    
    for net_idx, (models_net, nrn_models_net) in enumerate(zip(models, nrn_models)):
        for run_idx, ((features, model), (nrn_features, nrn_model)) in enumerate(zip(models_net, nrn_models_net)):
            _, sparsities[net_idx, run_idx] = getSparsity(model, bias_or_weight, print_sparsity=False)
            new_total_nodes[net_idx, run_idx] = get_new_total_nodes(features)
            _, total_sparsities[net_idx, run_idx] = getSparsity(nrn_model, bias_or_weight, print_sparsity=False)

    sparsities_frame_input = np.vstack((sparsities.flatten(), total_sparsities.flatten())).T

    sparsity_frame = getSparsitiesFrame(total_nodes, model_config.runs, np_array=sparsities_frame_input)
    add_new_index_chp(
        sparsity_frame, 
        new_total_nodes
    )
    
    method_str = get_method_str(method_config)
    saves_cls = Saves(save_directory=prune_path)
    saves_cls.savePrunFrames(sparsity_frame, method_str, 'sparsities.pkl')
    
    return sparsity_frame
    
    
    
    




if __name__ == '__main__':
    on_every_Prune_config_do(makeModelSparsities, add_folders=['_prun5',], only_add_folder=True)