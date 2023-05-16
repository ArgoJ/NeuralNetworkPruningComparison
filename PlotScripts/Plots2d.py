import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure


from DataClasses import OriginalFigure
from basicFrame import InTrainEvals

from PlotScripts.Plots import Plots
from PlotScripts.PlotHelpers import set_size




class Plots2d(Plots):
    def __init__(self) -> None:
        super().__init__()


    def plotModelData(
        self,
        original_data: tuple[np.ndarray,...],
        model_data: tuple[np.ndarray,...],
        train_data: tuple[np.ndarray,...],
        during_train_evals: InTrainEvals
    ) -> Figure:
        """Shows the loss and R² over epochs during training. 
        In a seperate subplot, it shows the original, model prediction and
        training datapoints. 

        INPUTS
        ------
        ``original_data`` is the original unprocessed linspace data. 
        ``model_data`` are the model predicted datapoints. 
        ``train_data`` is the unprocessed random training data. 
        ``loss`` is the loss during training of the training and validation data. 
        ``r_square`` is the R² Score during training of the training and validation data.

        RETURN
        ------
        ``fig`` is the figure with the loss, R² and model predictions.
        """
        loss = during_train_evals.xs(key='loss', axis=1, level=1)
        r2 = during_train_evals.xs(key='R²', axis=1, level=1)

        during_train_evals_len = len(during_train_evals)
        x = np.linspace(0, during_train_evals_len-1, during_train_evals_len)

        # plot figure
        fig = plt.figure(figsize=set_size((12, 10)), constrained_layout = True)

        ax_fun = fig.add_subplot(3, 1, 1)
        ax_fun.grid(True)
        ax_fun.set_xlabel('x') 
        ax_fun.set_ylabel('y')
        ax_fun.plot(*original_data, label='ground truth')
        ax_fun.plot(*model_data, label='prediction')
        ax_fun.plot(train_data[0][:, 0], train_data[1], 'o', label='traindata', markersize=2)
        ax_fun.legend(bbox_to_anchor=(1.01, 0.92), loc='lower right', ncol=3, frameon=False)

        ax_loss = fig.add_subplot(3, 1, 2)
        self.makeAxLoss(ax_loss, x, loss)
        ax_loss.tick_params(axis='x', which='both', labelbottom=False)

        ax_r2 = fig.add_subplot(3, 1, 3)
        self.makeAxR2(ax_r2, x, r2, sharex_ax=ax_loss)

        # legend
        handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
        ax_loss.legend(handles_loss, labels_loss, bbox_to_anchor=(1.01, 0.92), loc='lower right', ncol=2, frameon=False)

        plt.close()
        return fig


    def plotOrigFun(
        self, 
        original_data: tuple[np.ndarray,...]
    ) -> OriginalFigure:
        """Shows a plot of the ground truth. 

        INPUTS
        ------
        ``original_data`` is the ground truth data.  

        RETURN
        ------
        ``fig`` is the figure with the ground truth. 
        """
        fig, _ = plt.subplots(FigureClass=OriginalFigure, figsize=set_size((10,3)), layout="constrained")
        plt.xlabel("x") 
        plt.ylabel("y")
        plt.grid(True)
        plt.plot(*original_data, label='ground truth')
        # plt.legend(bbox_to_anchor=(1.01, 0.95), loc='lower right', frameon=False)

        plt.close()
        return fig