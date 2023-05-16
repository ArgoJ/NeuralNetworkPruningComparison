import sys
import os


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from Saves import Saves
from Loads import Loadings, loadConfig, getSavesDirectory
from Helpers import selectDevice, inputParams
from CustomExceptions import DimensionException

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from PerformanceScripts.MultiPerf import getPerformancesPruned, getPerformancesOriginal



def mainPerformanceInputs():
    device = selectDevice()
    dimension, time_folder = inputParams()
    return getSavesDirectory(dimension, time_folder)


def mainPerformance(save_directory: str, always_save = False, save_add_dirs: list = []):
    load_cls = Loadings(save_directory)
    config = load_cls.loadBaseConfig()

    # chooses if 2d or 3d
    if config.inputs==1:
        data_cls = Data2d(config)
    elif config.inputs==2:
        data_cls = Data3d(config)
    else:
        raise(DimensionException(config.inputs))

    save_performances = []
    prun_performance_tuple = getPerformancesPruned(load_cls, data_cls, config, base_dirs=[save_directory,*save_add_dirs])
    save_performances.append((*prun_performance_tuple, 'pruned_'))

    # orig_performance_path = os.path.join(loadings.getFramesDirectory(),  'performances.pkl')
    # orig_mean_performance_path = os.path.join(loadings.getFramesDirectory(),  'mean_performances.pkl')
    # if not os.path.exists(orig_performance_path) and not os.path.exists(orig_mean_performance_path):
    orig_performance_tuple = getPerformancesOriginal(load_cls, data_cls, config)
    save_performances.append(orig_performance_tuple)


    save = 'y' if always_save else input('To save data, type in \'y\'.\n')
    if 'y' in save:
        saves_cls = Saves(save_directory=save_directory)
        saves_cls.saveEveryPerformance(
            *save_performances
        )
        
        
def newPrunFolderPerformance(save_directory: str, always_save = False, base_dirs: list = []):
    load_cls = Loadings(save_directory)
    config = load_cls.loadBaseConfig()

    # chooses if 2d or 3d
    if config.inputs==1:
        data_cls = Data2d(config)
    elif config.inputs==2:
        data_cls = Data3d(config)
    else:
        raise(DimensionException(config.inputs))

    prun_performance_tuple = getPerformancesPruned(load_cls, data_cls, config, base_dirs=base_dirs)

    save = 'y' if always_save else input('To save data, type in \'y\'.\n')
    if 'y' in save:
        saves_cls = Saves(save_directory=save_directory)
        saves_cls.saveEveryPerformance(
            (*prun_performance_tuple, 'pruned_'), 
        )



if __name__ == "__main__": 
    save_directory = mainPerformanceInputs()
    mainPerformance(save_directory)