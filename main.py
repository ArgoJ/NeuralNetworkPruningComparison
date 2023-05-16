import os
from pathlib import Path


from Loads import loadArchitectureConfig, loadBasicDimConfig, makeConfigDataOfOthers, loadBaseMethodConfig, loadConfig
from Helpers import selectDevice, set_multi_thread, del_multi_thread
from mainModels import mainModels
from CustomExceptions import ignore_warning

from Pruning.mainPrun import mainPrun

from PerformanceScripts.mainPerformance import mainPerformance, newPrunFolderPerformance





def mainAllModels(config_dir: Path):
    """Makes all models of the configs in the 'Scripts/Configs/ArchitectureConfigs' folder 
    combined with the 'Scripts/Configs/BaseConfigs' folder. 
    The Script saves the frames and models after every finished configuration 
    in a {timefolder} in 'Saves2D' and 'Saves3D'.

    INPUTS
    ------
        ``config_dir`` is the directory of the configurations.
    """
    print('='*120 + '\n' + 'BASE MODELS')

    # base configs
    base_config_dir = os.path.join(config_dir, 'BaseConfigs')
    if os.path.exists(base_config_dir):
        base_config_names = os.listdir(base_config_dir)
    else:
        raise(NameError('No such path exists!', base_config_dir))

    # architecture configs
    architecture_config_dir = os.path.join(config_dir, 'ArchitectureConfigs')
    architecture_config_names = os.listdir(architecture_config_dir)

    for base_config_name in base_config_names:
        base_config = loadBasicDimConfig(os.path.join(base_config_dir, base_config_name))

        for architecture_config_name in architecture_config_names:
            architecture_config = loadArchitectureConfig(os.path.join(architecture_config_dir, architecture_config_name))

            config = makeConfigDataOfOthers(base_config, architecture_config)
            mainModels(config, always_save=True)



def mainAllPrunings(curr_dir: Path, config_dir: Path, ignore_folders = False, prun_dir_index: int = 0):
    """Pruns every {timefolder} in the 'Saves2D' and 'Saves3D' folder, 
    with every pruning method provided in the 'Scripts/Configs/PrunMethodConfigs' 
    and 'Scripts/Configs/PrunTypeConfigs'. 
    Saves the frames and models. 

    INPUTS
    ------
        ``curr_dir`` is the base directory for the 3D and 2D savings. 

        ``config_dir`` is the directory of the configurations.  
    """
    print('='*120 + '\n' + 'PRUNING')

    # method configs
    method_config_dir = os.path.join(config_dir, 'PrunMethodConfigs')
    if os.path.exists(method_config_dir):
        method_config_names = os.listdir(method_config_dir)

    # type configs
    type_config_dir = os.path.join(config_dir, 'PrunTypeConfigs')
    if os.path.exists(type_config_dir):
        type_config_names = os.listdir(type_config_dir)

    for dimension in [2, 3]:
        dim_load_dir = os.path.join(curr_dir, f'{dimension}D_Saves') 
        all_time_folders = os.listdir(dim_load_dir)
        dim_save_dir = dim_load_dir if prun_dir_index==0 else dim_load_dir + f'_prun{prun_dir_index}'

        for time_folder in all_time_folders:
            if ignore_folders and (time_folder == "02_16_13_03" or time_folder == "02_16_14_31" or time_folder == "02_16_17_41" or time_folder == "02_16_20_25" or time_folder == "02_28_09_58"):
                continue
            load_directory = os.path.join(dim_load_dir, time_folder)
            config = loadConfig(os.path.join(load_directory, 'config.json'))

            if 'method_config_names' in locals():
                for method_config_name in method_config_names:
                    print('='*110 + '\n' + f'DIM: {dimension}, FOLDER: {time_folder}, METHOD: {method_config_name}')
                    method_config = loadBaseMethodConfig(config, os.path.join(method_config_dir, method_config_name))
                    mainPrun(
                        load_directory, 
                        method_config, 
                        always_save=True, 
                        prun_save_dir=os.path.join(dim_save_dir, time_folder),
                    )

            if 'type_config_names' in locals():
                for type_config_name in type_config_names:
                    print('='*110 + '\n' + f'DIM: {dimension}, FOLDER: {time_folder}, METHOD: {type_config_name}')
                    method_config = loadBaseMethodConfig(config, os.path.join(type_config_dir, type_config_name))
                    mainPrun(
                        load_directory, 
                        method_config, 
                        always_save=True, 
                        prun_save_dir=os.path.join(dim_save_dir, time_folder)
                    )



def mainAllPerformances(repo_dir: Path, add_str2base_dirs: list[str] = [], fun2do = mainPerformance):
    """Makes the performance of every single model for every {timefolder} 
    in the folder 'Save2D' and 'Save3D'. 
    Saves the performance frames in the folder 'Frame' of every {timefolder}.

    INPUTS
    ------
        ``curr_dir`` is the base directory for the 3D and 2D savings.   
    """
    print('='*120 + '\n' + 'PERFORMANCE')

    for dimension in [2, 3]:
        dim_save_dir = os.path.join(repo_dir, f'{dimension}D_Saves')
        all_time_folders = os.listdir(dim_save_dir)

        for time_folder in all_time_folders:
            print('='*110 + '\n' + f'DIM: {dimension}, FOLDER: {time_folder}')
            save_directory = os.path.join(dim_save_dir, time_folder)
            base_dirs = [os.path.join(dim_save_dir + str_ext, time_folder) for str_ext in add_str2base_dirs]
            fun2do(save_directory, always_save=True, base_dirs=base_dirs)
            



if __name__ == "__main__":
    ignore_warning('Initializing zero-element tensors is a no-op')

    CURR_DIR = os.path.dirname(__file__)
    REPO_DIR = os.path.dirname(CURR_DIR)
    config_dir = os.path.join(CURR_DIR, 'Configs') 
    config_dir_2 = os.path.join(CURR_DIR, 'Configs2')
    config_dir_3 = os.path.join(CURR_DIR, 'Configs3')
    config_dir_4 = os.path.join(CURR_DIR, 'Configs4')
    device = selectDevice()

    set_multi_thread()

    # mainAllModels(config_dir)
    # mainAllPrunings(REPO_DIR, config_dir_4, ignore_folders=False, prun_dir_index=5)
    mainAllPerformances(REPO_DIR, add_str2base_dirs=['_prun5'], fun2do=newPrunFolderPerformance) 
    #, add_base_dirs=[f'_prun{i}' for i in range(1, 5)])

    del_multi_thread()
