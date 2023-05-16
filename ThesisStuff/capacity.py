import torch
import pandas as pd
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt

from timeit import timeit

from thesisSaves import ThesisSaves

# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from PlotScripts.PlotHelpers import set_size, make_latex_fonts, adjust_fig_size_legend_above



def trainFun(x):
    return np.exp(-6*x + 5) + 1

def testFun(x):
    return trainFun(x) + 0.5*(x-1)**2 + 2


def double_arrow_w_text(ax: plt.Axes, mid_xypt: tuple[float, float], size: float, text: str):
    ax.annotate(
        '', 
        xy=(mid_xypt[0], mid_xypt[1] - size/2), 
        xytext=(mid_xypt[0], mid_xypt[1] + size/2), 
        arrowprops=dict(arrowstyle="<->")
    )
    ax.text(mid_xypt[0] + 0.1, mid_xypt[1], text, fontsize=10, va="center")



def mainCapacity():
    x = np.linspace(0, 5, 200)

    test_y = testFun(x)
    inputs = (
        ('Training Error', trainFun(x)),
        ('Test Error', test_y)
    )
    test_miny = np.min(test_y)
    test_minx = np.where(test_y == test_miny)[0][0]

    optim_x = x[test_minx]
    optim_capacity_y = np.array([0, 25])
    optim_capacity_x = np.array([optim_x for _ in range(2)])

    linestyles = ('-', '--', '-.', ':', (0, (1, 10)), (0, (3, 5, 1, 5, 1, 5)))
    fig, ax = plt.subplots(figsize=set_size((8, 3)), layout="constrained")

    ax.set_xlabel('Capacity')
    ax.set_ylabel('Error')

    ax.plot(optim_capacity_x, optim_capacity_y, 'k' + '-') # linestyles[3]

    for (name, y), linestyle in zip(inputs, linestyles):
        ax.plot(x, y, linestyle, label=name)
    legend = ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1.02, 0.95), frameon=False)

    ax.set_ylim(0, 25)
    ax.set_xlim(0, 5)

    ax.set_xticks([optim_x])
    ax.set_xticklabels(['Optimal Capacity'])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)

    double_arrow_w_text(ax, (3.5, 3.5), 4, 'Generalization gap')

    ax.text(optim_x + 0.1, 20, 'Overfitting', fontsize=10, va='center')
    ax.text(optim_x - 0.1, 20, 'Underfitting', fontsize=10, va='center', horizontalalignment='right')
    
    adjust_fig_size_legend_above(fig, legend)

    return [(fig, 'capacity')]






if __name__ == '__main__':
    make_latex_fonts()
    plots = mainCapacity()
    # plt.show()
    thesisSaves = ThesisSaves()
    thesisSaves.savePlots(*plots)