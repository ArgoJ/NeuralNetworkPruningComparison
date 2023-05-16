import sys
import os
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt

# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from PlotsCompare import ComparePerformancePlots
from Loads import Loadings, getSavesDirectory
from Helpers import selectDevice, inputParams
from Saves import Saves
from Constants import PERFORMANCE_NAMES

from PlotScripts.PlotHelpers import make_latex_fonts


def mainPlotPerformance():
    ## DEVICE SELECTION
    device = selectDevice()

    ## DIRECTORY AND CONFIG
    dimension, time_folder = inputParams()
    save_directory = getSavesDirectory(dimension, time_folder)

    loadings = Loadings(save_directory)
    compare_plots_cls = ComparePerformancePlots()

    orig_performances, multi_prun_performances = loadings.loadPerformances()
    print(f'original performances: \n {orig_performances}')
    print(f'pruned performances: \n {multi_prun_performances}')

    perf_figs = compare_plots_cls.plotAllPrunMethodsCompares(
        orig_performances.drop([*PERFORMANCE_NAMES[3:]], axis=1), 
        multi_prun_performances.drop([*PERFORMANCE_NAMES[3:]], axis=1)
    )

    # show all Plots
    plt.show(block=True)

    save = input('To save performance plots, type in \'y\'.\n')
    if 'y' in save:
        saves_cls = Saves(save_directory=save_directory)
        saves_cls.savePerformanceFigures(
            *perf_figs
        )


if __name__ == "__main__":
    make_latex_fonts()
    mainPlotPerformance()