import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure


from DataClasses import OriginalFigure
from basicFrame import InTrainEvals

from PlotScripts.Plots import Plots
from PlotScripts.PlotHelpers import default_colorbar, set_size





class Plots3d(Plots):
    def __init__(self) -> None:
        super().__init__()


    def plotModelData(
        self, 
        error_data: tuple[np.ndarray,...], 
        train_data: tuple[np.ndarray,...],
        during_train_evals: InTrainEvals
    ) -> Figure:
        """Shows the loss and R² over epochs during training. 
        In a seperate subplot, it shows the error of the original to the 
        model predictions. 

        INPUTS
        ------
        ``error_data`` is the error of original to model predictions. 
        ``train_data`` is the unprocessed random training data. 
        ``loss`` is the loss during training of the training and validation data. 
        ``r_square`` is the R² Score during training of the training and validation data.

        RETURN
        ------
        ``fig`` is the figure with the loss, R² and the error of original to model preds.
        """
        loss = during_train_evals.xs(key='loss', axis=1, level=1)
        r2 = during_train_evals.xs(key='R²', axis=1, level=1)

        during_train_evals_len = len(during_train_evals)
        x = np.linspace(0, during_train_evals_len-1, during_train_evals_len)

        # plot figure
        fig = plt.figure(figsize=set_size((16, 5)), constrained_layout = True)

        ax_error = fig.add_subplot(1, 2, 2)
        cont = ax_error.contourf(*error_data, 100, cmap='RdBu_r')
        ax_error.set_xlabel('x')
        ax_error.set_ylabel('y')
        ax_error.set_aspect('equal')

        default_colorbar(fig, ax_error, cont, 'error')

        # plot traindata
        train_plot = ax_error.plot(train_data[0], train_data[1], 'og', markersize=.5)
        ax_error.legend([train_plot[0]], ['traindata'], bbox_to_anchor=(1.07, 0.92), loc='lower right', frameon=False)
        
        # plot loss of train and validation data
        ax_loss = fig.add_subplot(2, 2, 1)
        self.makeAxLoss(ax_loss, x, loss)
        ax_loss.tick_params(axis='x', which='both', labelbottom=False)

        # plot R² of train and validation data
        ax_r2 = fig.add_subplot(2, 2, 3)
        self.makeAxR2(ax_r2, x, r2, sharex_ax=ax_loss)

        # legend
        handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
        ax_loss.legend(handles_loss, labels_loss, bbox_to_anchor=(1.03, 0.82), loc='lower right', ncol=2, frameon=False)
        
        plt.close()
        return fig


    def plotOrigFun(
        self, 
        original_data: tuple[np.ndarray,...]
    ) -> OriginalFigure:
        """Shows a plot of the ground truth in a contour colorbar plot. 
        
        INPUTS
        ------
        ``original_data`` is the ground truth data. 

        RETURN
        ------
        ``fig`` is the figure with the ground truth. 
        """
        fig = plt.figure(FigureClass=OriginalFigure, figsize=set_size((8, 6), fraction=0.4), constrained_layout=True)
        
        ax_fun = fig.add_subplot()
        cont = plt.contourf(*original_data, 100, cmap='RdBu_r')
        ax_fun.set_xlabel('x')
        ax_fun.set_ylabel('y')
        ax_fun.set_aspect('equal')

        default_colorbar(fig, ax_fun, cont, 'z')

        plt.close()
        return fig