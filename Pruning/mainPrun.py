import sys
import os
import matplotlib.pyplot as plt


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from Saves import Saves
from Loads import Loadings, getSavesDirectory, getPrunMethod
from Helpers import selectDevice, inputParams, set_multi_thread, del_multi_thread
from CustomExceptions import DimensionException
from DataClassesJSON import MethodPrunConfig, PrunConfig

from PlotScripts.PlotHelpers import make_latex_fonts
from PlotScripts.Plots2d import Plots2d
from PlotScripts.Plots3d import Plots3d

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Pruning.MultiPrun import loadModelAndPrune



def mainPrunInputs():
    device = selectDevice()
    dimension, time_folder = inputParams()
    save_directory = getSavesDirectory(dimension, time_folder)
    method_config = getPrunMethod(save_directory)
    return save_directory, method_config


def mainPrun(
    save_directory: str, 
    method_config: MethodPrunConfig | PrunConfig, 
    always_save = False,
    prun_save_dir: str = '',
):
    loadings_cls = Loadings(save_directory)
    loaded_config = loadings_cls.loadBaseConfig()

    # chooses if 2d or 3d
    if loaded_config.inputs==1:
        # 2D stuff
        data_cls = Data2d(loaded_config)
        plots_cls = Plots2d()
    elif loaded_config.inputs==2:
        # 3D stuff
        data_cls = Data3d(loaded_config)
        plots_cls = Plots3d()
    else:
        raise(DimensionException(loaded_config.inputs))

    multi_prun_net_data = loadModelAndPrune(
            loadings_cls, 
            data_cls,
            plots_cls, 
            loaded_config,  
            method_config
        )

    plt.show(block=True)


    save = 'y' if always_save else input('To save data, type in \'y\'.\n')
    if 'y' in save:
        saves_cls = Saves(save_directory=save_directory if not prun_save_dir else prun_save_dir)
        saves_cls.saveEverythingPrun(
            method_config, 
            *multi_prun_net_data
        )



if __name__ == "__main__":
    inputs = mainPrunInputs()
    make_latex_fonts()
    
    set_multi_thread()
    mainPrun(*inputs)
    del_multi_thread()