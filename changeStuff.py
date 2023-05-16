import torch
import pandas as pd
import os
import numpy as np
import json
import shutil


from pathlib import Path
from time import time
from timeit import timeit


from Saves import save_JSON, Saves
from Loads import getDimensionSaveDirectory, Loadings, loadConfig

from basicFrame import getMeanMetricsFrame
from mainEvals import makeModelEvals
from Helpers import (
    selectDevice, 
    getMeanFromRuns, 
    add_new_index_chp, 
    get_new_total_nodes, 
    createDir, 
    add_to_existing_frame_file, 
    find_new_pkl_path
)
from CustomExceptions import DimensionException
from DataClasses import Models
from Constants import MEAN_NAMES

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Pruning.PrunHelpers import getPrunConfig, removeNodes




#====================================================================================================
# ADD PARAMS TO METHOD CONFIG
#====================================================================================================
def add_param_to_PrunConfigs(fun):
    for i in range(2,4):
        base_dir = getDimensionSaveDirectory(i)
        arch_folders = os.listdir(base_dir)
        for arch_folder in arch_folders:
            arch_path = os.path.join(base_dir, arch_folder, 'Pruning')

            prun_folders = os.listdir(arch_path)
            for prun_folder in prun_folders:
                config_name = 'method_config.json'
                config_path = os.path.join(arch_path, prun_folder, config_name)

                with open(config_path) as json_file:
                    json_dict = json.load(json_file)
                new_dict = fun(json_dict)
                if new_dict is not None and json_dict!=new_dict:
                    save_JSON(new_dict, config_path)



def add_prun_prev_params(json_dict: dict):
    if 'iterations' in json_dict:
        return {
            "type": json_dict["type"],
            "networks": json_dict["networks"],
            "layer": json_dict["layer"],
            "nodes": json_dict["nodes"],
            "amount": json_dict["amount"],
            "bias_or_weight": json_dict["bias_or_weight"],
            "remove_nodes": json_dict["remove_nodes"],
            "prun_prev_params": True,
            "prun_next_params": json_dict["prun_next_params"],
            "iterations": json_dict["iterations"],
            "method": json_dict["method"],
        }
    else:
        return {
            "type": json_dict["type"],
            "networks": json_dict["networks"],
            "layer": json_dict["layer"],
            "nodes": json_dict["nodes"],
            "amount": json_dict["amount"],
            "bias_or_weight": json_dict["bias_or_weight"],
            "remove_nodes": json_dict["remove_nodes"],
            "prun_prev_params": True,
            "prun_next_params": json_dict["prun_next_params"],
        }
    


def add_prun_schedule_params(json_dict: dict):
    if 'iterations' in json_dict:
        return {
            "type": json_dict["type"],
            "networks": json_dict["networks"],
            "layer": json_dict["layer"],
            "nodes": json_dict["nodes"],
            "amount": json_dict["amount"],
            "bias_or_weight": json_dict["bias_or_weight"],
            "remove_nodes": json_dict["remove_nodes"],
            "prun_prev_params": json_dict["prun_prev_params"],
            "prun_next_params": json_dict["prun_next_params"],
            "iterations": json_dict["iterations"],
            "schedule": '' ,#if json_dict["iterations"]<=1 else 'binom',
            "method": json_dict["method"],
        }
    else:
        return json_dict
    
    

def add_prun_lnpp_params(json_dict: dict):
    if 'iterations' in json_dict:
        return {
            "type": json_dict["type"],
            "networks": json_dict["networks"],
            "layer": json_dict["layer"],
            "nodes": json_dict["nodes"],
            "amount": json_dict["amount"],
            "bias_or_weight": json_dict["bias_or_weight"],
            "remove_nodes": json_dict["remove_nodes"],
            "prun_prev_params": json_dict["prun_prev_params"],
            "prun_next_params": json_dict["prun_next_params"],
            "iterations": json_dict["iterations"],
            "schedule": json_dict["schedule"] ,#if json_dict["iterations"]<=1 else 'binom',
            "last_iter_npp": False,
            "method": json_dict["method"],
        }
    else:
        return json_dict




#====================================================================================================
# RENAME STR TO STR_PP
#====================================================================================================
def _rename_fun(name: str):
    return name.replace('str_rn', 'str_rn_pp') if 'str_rn' in name else name.replace('str', 'str_pp')


def _rename_illus(name: str):
    return name.replace('str_pp', 'str')


def change_prun_names(rename_fun):
    for i in range(2,4):
        base_dir = getDimensionSaveDirectory(i)
        arch_folders = os.listdir(base_dir)
        for arch_folder in arch_folders:
            arch_path = os.path.join(base_dir, arch_folder, 'Pruning')
            _change_names(arch_path, rename_fun, 'str_')





#====================================================================================================
# REDO MODEL WITH NRN MODEL
#====================================================================================================
def redo_pruned_models(base_path: Path, prune_path: Path):
    loaded_config = loadConfig(os.path.join(base_path, 'config.json'))

    
    load_cls = Loadings(base_path)
    prun_base_path = os.path.dirname(os.path.dirname(prune_path))
    load_cls.updateDirectorys(prun_base_path)
    prun_folder = os.path.basename(prune_path)
    load_cls.setPrunMethodDirectorys(prun_folder)

    method_config = load_cls.loadMethodConfig()
    model_config = getPrunConfig(loaded_config, method_config)
    
    models = load_cls.loadAllModels_ForPrun(model_config, loaded_config, use_notRemNode_models=True)
    new_models = Models([removeNodes(features, model) for features, model in models_net] for models_net in models)

    if loaded_config.inputs==1:
        # 2D stuff
        data_cls = Data2d(loaded_config)
    elif loaded_config.inputs==2:
        # 3D stuff
        data_cls = Data3d(loaded_config)
    else:
        raise(DimensionException(loaded_config.inputs + 1))
    
    new_metrics = makeModelEvals(data_cls, model_config, new_models)
    new_mean_metrics = getMeanFromRuns(new_metrics, getMeanMetricsFrame)


    new_total_nodes = np.zeros((model_config.networks, model_config.runs), dtype=np.int64)
    for net_idx, new_models_net in enumerate(new_models):
        for run_idx, (new_features, _) in enumerate(new_models_net):
            new_total_nodes[net_idx, run_idx] = get_new_total_nodes(new_features)
    # new index of metrics
    add_new_index_chp(
        new_metrics, 
        new_total_nodes
    )
    add_new_index_chp(
        new_mean_metrics, 
        np.tile(np.mean(new_total_nodes, axis=1), (len(MEAN_NAMES), 1)).T
    )

    saves_cls = Saves(save_directory=prun_base_path)
    saves_cls.saveEverythingPrun(method_config, new_models, new_metrics, new_mean_metrics)
    


#====================================================================================================
# REDO STUFF IN PRUNE CONFIGS AND BASE CONFIGS
#====================================================================================================
def on_every_Prune_config_do(function, add_folders: list = [], only_add_folder = False, **kwargs):
    outputs = []
    for dim in [2, 3]:
        dim_dir = getDimensionSaveDirectory(dim)
        all_arch_folders = os.listdir(dim_dir)

        arch_outputs = []
        for arch in all_arch_folders:
            base_path = os.path.join(dim_dir, arch)
            add_paths = [os.path.join(dim_dir + str_ext, arch, 'Pruning') for str_ext in add_folders]
        
            if only_add_folder:
                for add_path in add_paths:
                    prune_paths = [os.path.join(add_path, prun_folder) for prun_folder in os.listdir(add_path)]
            else:
                prun_path = os.path.join(base_path, 'Pruning')
                prun_folders = os.listdir(prun_path)
                prune_paths = [os.path.join(prun_path, prun_folder) for prun_folder in prun_folders]
                for add_path in add_paths:
                    prune_paths.extend([os.path.join(add_path, prun_folder) for prun_folder in os.listdir(add_path)])

            prune_outputs = []
            for prune_path in prune_paths:
                prune_str = os.path.basename(prune_path)
                print(f'DIM: {dim}, FOLDER: {arch}, METHOD: {prune_str}')
                output = function(base_path, prune_path, **kwargs)
                
                prune_outputs.append((prune_str, output))
            arch_outputs.append((arch, prune_outputs))
        outputs.append((dim, arch_outputs))
    return outputs



def on_every_base_config_do(function, **kwargs):
    outputs = []
    for dim in [2, 3]:
        dim_dir = getDimensionSaveDirectory(dim)
        all_arch_folders = os.listdir(dim_dir)

        arch_outputs = []
        for arch_folder in all_arch_folders:
            base_path = os.path.join(dim_dir, arch_folder)
            output = function(dim, base_path, **kwargs)
            arch_outputs.append((arch_folder, output))
        outputs.append((dim, arch_outputs))
    return outputs





#====================================================================================================
# RENAME MODEL TO OLD MODEL
#====================================================================================================
def _rename_OldModels(name: str):
    return name.replace('models', 'old_models')


def change_pruneStuff(function, *args, add_folders = []):
    for dimension in [2, 3]:
        dim_dir = getDimensionSaveDirectory(dimension)
        all_arch_folders = os.listdir(dim_dir)

        for arch_folder in all_arch_folders:
            base_path = os.path.join(dim_dir, arch_folder)
            prun_path = os.path.join(base_path, 'Pruning')
            add_paths = [os.path.join(dim_dir + str_ext, arch_folder, 'Pruning') for str_ext in add_folders]
            prun_folders = os.listdir(prun_path)
            prune_paths = [os.path.join(prun_path, prun_folder) for prun_folder in prun_folders]

            for add_path in add_paths:
                prune_paths.extend([os.path.join(add_path, prun_folder) for prun_folder in os.listdir(add_path)])

            for prune_path in prune_paths:
                function(prune_path, *args)



            

#====================================================================================================
# RENAME ANYTHING
#====================================================================================================
def _change_names(base_path: str, rename_fun, str2search: str, startsWith: bool = False):
    for root, dirs, files in os.walk(base_path):
        
        # files in the current folder
        for file in files:
            if (str2search in file and not startsWith) or (file.startswith(str2search) and startsWith):
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, rename_fun(file))
                os.rename(old_file_path, new_file_path)

        # subfolders in the current folder
        for dir in dirs:
            if (str2search in dir and not startsWith) or (dir.startswith(str2search) and startsWith):
                old_dir_path = os.path.join(root, dir)
                new_dir_path = os.path.join(root, rename_fun(dir))
                os.rename(old_dir_path, new_dir_path)
                _change_names(new_dir_path, rename_fun, str2search)


def _delete_files(base_path: str, str2search: str, startsWith: bool = False):
    for root, dirs, files in os.walk(base_path):
        
        # files in the current folder
        for file in files:
            if (str2search in file and not startsWith) or (file.startswith(str2search) and startsWith):
                old_file_path = os.path.join(root, file)
                os.remove(old_file_path)

        # subfolders in the current folder
        for dir in dirs:
            if (str2search in dir and not startsWith) or (dir.startswith(str2search) and startsWith):
                old_dir_path = os.path.join(root, dir)
                _delete_files(old_dir_path, str2search)



def copy_prun_files(base_path: Path, prune_path: Path, new_path:Path, following_path: Path):
    old_file_path = os.path.join(prune_path, following_path)

    split_paths = os.path.normpath(prune_path).split(os.path.sep)
    between_path = os.path.join(*split_paths[-4:])
    new_file_path = os.path.join(new_path, between_path, following_path)
    createDir(os.path.dirname(new_file_path))
    shutil.copy(old_file_path, new_file_path)



def insert_perf_frame(dim: int, base_path: Path, mid_path: str):
    dim_path, time_str = os.path.split(base_path)
    repo_path, dim_str = os.path.split(dim_path)
    loading_base_path = os.path.join(repo_path, mid_path, dim_str, time_str)
    load_cls = Loadings(directory=loading_base_path)
    
    insert_perf, inser_mean_perf = load_cls.loadPrunPerformances()
    main_perf_path = os.path.join(base_path, 'Frames', 'pruned_performances.pkl')
    main_mean_perf_path = os.path.join(base_path, 'Frames', 'pruned_mean_performances.pkl')
    
    methods_perf_len = insert_perf.index.droplevel(level=(7, 8)).unique().size
    methods_mean_perf_len = inser_mean_perf.index.droplevel(level=(7, 8)).unique().size
    
    add_to_existing_frame_file(insert_perf, main_perf_path)
    add_to_existing_frame_file(inser_mean_perf, main_mean_perf_path)
    
    
    

def add_lnp_to_prun_perf_methods(dim: int, base_path: Path):
    load_cls = Loadings(directory=base_path)
    
    perf, mean_perf = load_cls.loadPrunPerformances()
    change_perf_pkl_lnp(perf, 'pruned_performances.pkl', base_path)
    change_perf_pkl_lnp(mean_perf, 'pruned_mean_performances.pkl', base_path)
    
    
def change_perf_pkl_lnp(perf_df: pd.DataFrame, file_name: str, base_path: Path):   
    old_index = perf_df.index
    level_node_prun = old_index.get_level_values(level='node_prun')
    level_iter = old_index.get_level_values(level='iterations')
    level_type = old_index.get_level_values(level='type')
    
    new_level_node_prun = [
        prun_node + 'lnp_' if (5==prun_iter and '_str' in prun_type) else prun_node for prun_iter, prun_type, prun_node in zip(level_iter, level_type, level_node_prun)
        ]
    perf_df.index = pd.MultiIndex.from_arrays(
        [
            old_index.get_level_values(level=0), 
            old_index.get_level_values(level=1), 
            old_index.get_level_values(level=2), 
            old_index.get_level_values(level=3), 
            new_level_node_prun,
            old_index.get_level_values(level=5), 
            old_index.get_level_values(level=6), 
            old_index.get_level_values(level=7), 
            old_index.get_level_values(level=8), 
        ], 
        names=old_index.names
    )
    methods_perf_len = perf_df.index.droplevel(level=(7, 8)).unique().size

    
    new_file_path = os.path.join(base_path, 'Frames', file_name)
    if os.path.exists(new_file_path):
        loaded_frame: pd.DataFrame = pd.read_pickle(new_file_path)
        old_file_path = os.path.join(base_path, 'Frames', f'old_{file_name}')
        old_file_path = find_new_pkl_path(old_file_path)
        loaded_frame.to_pickle(old_file_path)
    perf_df.to_pickle(new_file_path)
    
    






if __name__ == '__main__':
    # add_param_to_PrunConfigs(add_prun_prev_params)
    # add_param_to_PrunConfigs(add_prun_schedule_params)
    # add_param_to_PrunConfigs(add_prun_lnpp_params)
    # change_prun_names(_rename_fun)

    # change_pruneStuff(_change_names, _rename_OldModels, 'models_', True, add_folders=[f'_prun{i}' for i in range(1, 3)])
    # change_pruneStuff(_delete_files, 'old_models_', True, add_folders=[f'_prun{i}' for i in range(1, 3)])

    # on_every_Prune_config_do(redo_pruned_models, [f'_prun{i}' for i in range(1, 3)])
    # on_every_Prune_config_do(
    #     copy_prun_files, 
    #     new_path=input('Type in the folder, where you want to save the pruning stuff!\n').replace('"', ''), 
    #     following_path=os.path.join('Frames', 'sparsities.pkl'))
    
    # on_every_base_config_do(insert_perf_frame, mid_path='all_results_Apr28')
    # on_every_base_config_do(add_lnp_to_prun_perf_methods)
    pass
