import sys
import os
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from MultiNets import plotAllMultiNets
from Saves import Saves
from Helpers import selectDevice
from CustomExceptions import DimensionException

from PlotScripts.Plots2d import Plots2d
from PlotScripts.Plots3d import Plots3d
from PlotScripts.PlotHelpers import make_latex_fonts



def mainPlots():
    ## DEVICE SELECTION
    device = selectDevice()


    ## DIRECTORY AND CONFIG
    dimension = int(input('Choose a dimension (2 or 3)!\n'))

    # chooses if 2d or 3d
    if dimension==2:
        # 2D stuff
        plots_cls = Plots2d()
    elif dimension==3:
        # 3D stuff
        plots_cls = Plots3d()
    else:
        raise(DimensionException(dimension))

    
    plotAllMultiNets(plots_cls, dimension, one_run_modelPlot=True, save_directly=True)



if __name__ == "__main__":
    make_latex_fonts()
    mainPlots()