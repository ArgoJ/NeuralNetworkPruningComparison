import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

from matplotlib.ticker import ScalarFormatter
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap


from Constants import PLOT_HEIGHT, PLOT_STYLES
from Helpers import largest_multiplier
from DataClasses import PerformanceFigures

from PlotScripts.PlotHelpers import get_plot_lim, get_unique_level_values, set_size, use_ScalarFormater





class ComparePerformancePlots:
    def __init__(self) -> None:
        pass


    def plotCompPrunMethodWithOrig(
        self,
        orig_performances: pd.DataFrame,
        prun_performances: pd.DataFrame,
        changed_hyperparam: tuple[str, int]
    ) -> Figure:
        performance_names = orig_performances.columns.get_level_values(0).unique()
        runs = prun_performances.index.get_level_values('run').unique()
        prun_methods = get_unique_level_values(prun_performances)

        plot_len = len(performance_names) + 1
        if not (plot_len/2).is_integer():
            plot_len += 1
        subplot_width = largest_multiplier(plot_len)
        subplot_height = int(plot_len/subplot_width)
        i_t = None


        fig, axs = plt.subplots(
            subplot_height, 
            subplot_width, 
            figsize=set_size((PLOT_HEIGHT*subplot_width, PLOT_HEIGHT*subplot_height)), 
            layout="constrained"
        )
        # fig.suptitle(f'{changed_hyperparam[1]} {changed_hyperparam[0]}')

        for i, perf_name in enumerate(performance_names):
            orig_perf = orig_performances[perf_name].to_numpy()
            prun_perf = prun_performances[perf_name]

            unique_orig = np.unique(orig_perf)

            ax_row = i // subplot_width
            ax_col = i % subplot_width
            if ax_col==(subplot_width-1) and ax_row==0:
                legend_ax = axs[ax_row, ax_col]
                legend_ax.axis('off')
                i += 1
            elif ax_row>0:
                i += 1
            ax = axs[i // subplot_width, i % subplot_width]
            
            if len(unique_orig) == 1:
                unique_prun = [prun_perf.loc[(*method, runs)] for method in prun_methods]
                self.blockSubplot(
                    ax,
                    *calcMeanYerr(unique_prun),
                    unique_orig,
                    perf_name
                )
            else:
                self.scatterSubplot(
                    ax, 
                    prun_perf, 
                    orig_perf,
                    runs, 
                    prun_methods,
                    perf_name
                )

            if 'time' in perf_name:
                i_t = i
        handles, labels = axs[i_t // subplot_width, i_t % subplot_width].get_legend_handles_labels()
        # legend_ax.legend(handles=handles, labels=labels, loc='upper right')


        return fig


    def blockSubplot(
        self, 
        ax: plt.Axes, 
        prun_means, 
        prun_yerrs,
        orig_data, 
        perf_name: str
    ) -> plt.Axes:
        bars = np.concatenate((orig_data, prun_means))
        yerrs = np.concatenate(([[0, 0]], prun_yerrs)).T
        bars_str = ['original', *['' for _ in range(len(bars) - 1)]]
        cmap = cm.get_cmap('tab20') 
        colors = list(cmap(range(len(bars))))
        y_pos = list(range(len(bars)))
        ax.bar(y_pos, bars, color=colors, yerr=yerrs)
        ax.set_ylabel(perf_name)
        ax.set_xticks(y_pos, bars_str)

        use_ScalarFormater(ax.yaxis)
        return ax
        

    def scatterSubplot(
        self, 
        ax: plt.Axes, 
        prun_data: pd.Series, 
        orig_data: pd.Series, 
        runs: pd.Series, 
        prun_methods, 
        perf_name,
    ) -> plt.Axes:
        min_plot, max_plot = get_plot_lim(
            np.concatenate((orig_data, prun_data.to_numpy()))
        )
        ax.grid()
        ax.set_aspect('equal')
        if min_plot!=max_plot:
            ax.set_ylim([min_plot, max_plot]) 
            ax.set_xlim([min_plot, max_plot]) 
            lin = [min_plot, max_plot]
            ax.plot(lin, lin, 'tab:gray', linewidth=0.5)
        ax.set_xlabel(f'{perf_name}(Unpr)')
        ax.set_ylabel(f'{perf_name}(Pr)')

        cmap = cm.get_cmap('tab20') 
        for k, (c, method) in enumerate(zip(cmap(range(len(prun_methods))), prun_methods)):
            method_str = self._make_method_str(method)
            y = prun_data.loc[(*method, runs)].to_numpy()
            ax.scatter(
                orig_data, 
                y, 
                marker=PLOT_STYLES[k % len(PLOT_STYLES)], 
                color=c,
                s=8,
                label=method_str
            )
        return ax


    def plotAllPrunMethodsCompares(
        self,
        orig_performances: pd.DataFrame,
        multi_prun_performances: pd.DataFrame
    ) -> tuple[PerformanceFigures, pd.Index]: 
        changed_hyperparams = orig_performances.index.get_level_values(0).unique()
        changed_hyperparam_name = changed_hyperparams.name

        figures = PerformanceFigures([])
        for changed_hyperparam in changed_hyperparams:
            fig = self.plotCompPrunMethodWithOrig(
                orig_performances.xs(changed_hyperparam, level=0), 
                multi_prun_performances.xs(changed_hyperparam, level=7), # nodes or layer
                (changed_hyperparam_name, changed_hyperparam)
            )
            figures.append(fig)
        return figures, changed_hyperparams

    
    def _make_method_str(self, method):
        method_seperator = ' '
        return method_seperator.join([str(obj) for obj in method if pd.notnull(obj)])




def calcMeanYerr(perfs: list[pd.Series]):
    means = []
    yerrs = []
    for perf in perfs:
        mean_p = perf[0] if all(perf[0] == perf) else perf.mean()
        min_p = perf.min()
        max_p = perf.max()

        means.append(mean_p)
        yerrs.append([mean_p - min_p, max_p - mean_p])
    return means, yerrs