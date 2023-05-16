import sys
import os
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Saves import Saves
from Loads import Loadings, loadConfig, getSavesDirectory
from Helpers import selectDevice, inputParams
from CustomExceptions import DimensionException

from Pruning.MultiPrun import plotAllPrunedFolders

from PlotScripts.Plots2d import Plots2d
from PlotScripts.Plots3d import Plots3d
from PlotScripts.PlotHelpers import make_latex_fonts



def mainPlotsPrun():
    ## DEVICE SELECTION
    device = selectDevice()


    ## DIRECTORY AND CONFIG
    dimension, time_folder = inputParams()
    save_directory = getSavesDirectory(dimension, time_folder)
    loaded_config = loadConfig(os.path.join(save_directory, 'config.json'))
    loadings = Loadings(save_directory)


    # chooses if 2d or 3d
    if loaded_config.inputs==1:
        # 2D stuff
        plots_cls = Plots2d()
    elif loaded_config.inputs==2:
        # 3D stuff
        plots_cls = Plots3d()
    else:
        raise(DimensionException(loaded_config.inputs))

    plotAllPrunedFolders(
        loadings, 
        plots_cls, 
        loaded_config,
        one_run=True,
        save_directly=True
    )



if __name__ == "__main__":
    make_latex_fonts()
    mainPlotsPrun()