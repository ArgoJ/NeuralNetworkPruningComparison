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




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2 / (1 + np.exp(-2*x)) - 1

def reLU(x):
    return np.maximum(0, x)

def leakyReLU(x, alpha: float = 0.1):
    return np.maximum(alpha*x, x)


def diffSigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def diffTanh(x):
    return 1 - tanh(x)**2

def diffReLU(x):
    return np.where(x >= 0, 1, 0)

def diffLeakyReLU(x, alpha: float = 0.1):
    return np.where(x >= 0, 1, alpha)



def mainActFunsPlot():
    loop = 10000
    x = np.linspace(-5, 5, 200)

    # time1 = timeit(lambda: leakyReLU(x), globals=globals(), number=loop)
    # time2 = timeit(lambda: diffleakyReLU(x), globals=globals(), number=loop)
    # print(f'maximum: {time1}')
    # print(f'where: {time2}')

    plot_inputs = (
        ('Sigmoid', sigmoid(x)),
        ('Tanh', tanh(x)),
        ('ReLU', reLU(x)),
        ('Leaky ReLU', leakyReLU(x)),
    )

    plot_diff_inputs = (
        ('Sigmoid', diffSigmoid(x)),
        ('Tanh', diffTanh(x)),
        ('ReLU', diffReLU(x)),
        ('Leaky ReLU', diffLeakyReLU(x)),
    )

    linestyles = ('-', '--', '-.', ':', (0, (1, 10)), (0, (3, 5, 1, 5, 1, 5)))
    figures = []
    for title, inputs, file_name, ylabel in zip(
        ['Activation functions', 'Derivatives for activation functions'], 
        [plot_inputs, plot_diff_inputs], 
        ['act_funs', 'act_funs_diff'],
        [r'$\sigma \left(z \right)$', r'$\sigma^{\prime} \left(z \right)$']
    ):
        fig, ax = plt.subplots(figsize=set_size((4, 3), fraction=0.47), layout="constrained")

        ax.grid()
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(ylabel)

        for (name, y), linestyle in zip(inputs, linestyles):
            ax.plot(x, y, linestyle, label=name)
        legend = ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1.02, 0.95), frameon=False)
        adjust_fig_size_legend_above(fig, legend)

        figures.append((fig, file_name))
    return figures
        
         



if __name__ == '__main__':
    make_latex_fonts()
    plots = mainActFunsPlot()
    # plt.show()
    thesisSaves = ThesisSaves()
    thesisSaves.savePlots(*plots)
    