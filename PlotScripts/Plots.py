import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from matplotlib.figure import Figure
from timeit import timeit
from itertools import cycle



from Constants import (
    METRIC_NAMES, 
    MEAN_NAMES, 
    ENVIROMENT_NAMES, 
    PLOT_HEIGHT, 
    WIDE_PLOT_WIDTH, 
    MULTIPLIER_WIDTH_COLORBAR, 
    PLOT_STYLES,
)
from Helpers import largest_multiplier
from DataClasses import MetricFigures, IterPrunFigures
from DataClassesJSON import PrunConfig, MethodPrunConfig
from Saves import get_method_str

from PlotScripts.PlotHelpers import *





class Plots(ABC):
    def __init__(self) -> None:
        self.incl_prun_envs = ['train', 'test', 'train pruned', 'test pruned']
        self.selected_envs = (ENVIROMENT_NAMES[1], )
        self.selected_metric_names = (METRIC_NAMES[0], )


    def plotAllRunMetrics(
        self,
        metrics: pd.DataFrame
    ) -> MetricFigures:
        """Plots for every run the metrics where one can see the behavior of models with different hyperparams. 

        INPUTS
        ------
            ``metrics`` 

        RETURN
        ------
            ``metrics_figs`` are the figures of every run with the metrics of multimodels. 

        """
        metrics_figs = MetricFigures()
        for _, run_metrics in metrics.groupby(level=1):
            fig_metrics = self.plotMetrics(run_metrics)
            metrics_figs.append(fig_metrics)
        return metrics_figs


    def makeAxLoss(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        loss: pd.DataFrame,
        sharex_ax: plt.Axes = None
    ) -> None:
        ax.grid(True)
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        if sharex_ax is not None:
            ax.sharex(sharex_ax)
        for col in loss:
            loss_data = loss[col]
            ax.plot(x, loss_data, label=loss_data.name)


    def makeAxR2(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        r2: pd.DataFrame,
        sharex_ax: plt.Axes = None
    ) -> None:
        ax.grid(True)
        ax.set_xlabel('epoch') 
        ax.set_ylabel('R²')
        ax.set_ylim([-0.05, 1.05])
        if sharex_ax is not None:
            ax.sharex(sharex_ax)
        for col in r2:
            loss_data = r2[col]
            ax.plot(x, loss_data, label=loss_data.name)


    def plotMetrics(
        self,
        metrics: pd.DataFrame
    ) -> Figure:
        """Shows a plot of the metrics R² Score, RMSE and MAE of a single run and multi networks.

        INPUTS
        ------
            ``metrics`` 

        RETURN
        ------
            ``fig`` is the figure of the mean metrics R² Score, RMSE and MAE.
        """
        col_length = len(self.selected_metric_names)
        
        # plot figure
        fig, axs = plt.subplots(col_length, 1, figsize=set_size((WIDE_PLOT_WIDTH, PLOT_HEIGHT*col_length)), layout="constrained", sharex=True)

        x_data = metrics.index.get_level_values(0)
        

        for i, metric_name in enumerate(self.selected_metric_names):
            ax = axs[i] if type(axs)==list else axs
            ax.grid(True)
            ax.set_ylabel(metric_name)
            if metric_name=='R²': 
                ax.set_ylim([-0.05, 1.05]) 
            metric = metrics.xs(metric_name, level=1, axis=1)

            for k, env_name in enumerate(ENVIROMENT_NAMES):
                y_data = metric[env_name].to_list()
                ax.plot(x_data, y_data, f'{PLOT_STYLES[k]}-', label=env_name)
            
            if i == 0:
                ax.legend(bbox_to_anchor=(1.01, 0.95), loc='lower right', ncol=2, frameon=False)
            if i == (col_length-1):
                x_label = x_data.name
                ax.set_xlabel(x_label) 

        plt.close()
        return fig


    def plotMeanMetrics(
        self,
        *multi_metrics: pd.DataFrame,
        x_level_index: int = 0,
        sparsity_plot = False,
    ) -> Figure:
        """Shows a errorbar plot of the mean metrics R² Score, RMSE and MAE of 
        multi runs and multi networks.

        INPUTS
        ------
        ``metrics_hyper_tuples`` can be multiple tuples with a list of the 
        train mean metrics with it's errors, a list of the test mean metrics 
        with it's errors and the associated hyperparameter frame. 

        RETURN
        ------
        ``fig`` is the figure of the mean metrics R² Score, RMSE and MAE.
        """
        
        col_length = len(self.selected_metric_names)
        
        if sparsity_plot:
            plot_height = 4
            plot_width = 8
        else:
            plot_height = PLOT_HEIGHT*col_length
            plot_width = WIDE_PLOT_WIDTH
        
        # plot figure
        fig, axs = plt.subplots(
            col_length, 1, 
            figsize=set_size((plot_width, plot_height)), 
            layout="constrained", 
            sharex=True
        )

        for k, metrics in enumerate(multi_metrics):
            x_data = metrics.index.get_level_values(x_level_index)[::3]

            mean_idx = metrics.index.isin([MEAN_NAMES[0]], level='means')
            min_idx = metrics.index.isin([MEAN_NAMES[1]], level='means')
            max_idx = metrics.index.isin([MEAN_NAMES[2]], level='means')

            for i, metric_name in enumerate(self.selected_metric_names):
                ax = axs[i] if type(axs)==list else axs
                ax.grid(True)
                ax.set_ylabel(metric_name)
                if metric_name=='R²': 
                    ax.set_ylim([-0.05, 1.05]) 
                metric = metrics.xs(metric_name, level=1, axis=1)
                
                for l, env_name in enumerate(ENVIROMENT_NAMES):
                    env_metric = metric[env_name]
                    y_data = env_metric[mean_idx].to_list()
                    y_error = [env_metric[min_idx].to_list(), env_metric[max_idx].to_list()]
                    
                    style_index = k*2 + l
                    ax.errorbar(
                        x_data, 
                        y_data, 
                        yerr=y_error, 
                        fmt=f'{PLOT_STYLES[style_index]}-', 
                        capsize=5, 
                        elinewidth=1.5, 
                        label=self.incl_prun_envs[style_index]
                    )
                if i == 0 and sparsity_plot:
                    ax.legend(bbox_to_anchor=(1.01, 1.1), loc='lower right', ncol=2*len(multi_metrics), frameon=False)   
                if i == 0 and not sparsity_plot:
                    ax.legend(bbox_to_anchor=(1.01, 0.95), loc='lower right', ncol=2*len(multi_metrics), frameon=False)
                if i == (col_length-1):
                    last_ax = ax
                    last_ax.set_xlabel(x_data.name)
                    if sparsity_plot:
                        set_xticks_int_only(last_ax)

        plt.close()
        return fig


    def firstPlot(
        self,
        ax: plt.Axes,
        unpruned_metric: pd.Series,
        prun_metric: pd.Series,
        old_run_chp: np.ndarray
    ):
        chp_len = len(old_run_chp)
        cmap = chooseCmap(chp_len) 
        for (_, xp), (_, yp), m in zip(
            unpruned_metric.groupby(level='run'), 
            prun_metric.groupby(level='run'),  
            PLOT_STYLES
        ):
            pl = ax.scatter(xp, yp, c=range(chp_len), cmap=cmap, marker='.', s=5, vmin=-0.5, vmax=chp_len-0.5, alpha=0.7)
        return pl


    def secondPlot_top(
        self,
        ax: plt.Axes,
        survived_nets: list[list],
        old_run_chp: np.ndarray,
        colors: list,
        box_width: float = 0.8  # default box width
    ): 
        flierprops = dict(marker='.', markerfacecolor='k', markersize=2, linestyle='none')
        box_plt = ax.boxplot(survived_nets, positions=old_run_chp, patch_artist=True, widths=box_width, flierprops=flierprops)
        for patch, color in zip(box_plt['boxes'], colors):
            patch.set_facecolor(color)
        return box_plt
    

    def secondPlot_bottom(
        self,
        ax: plt.Axes,
        runs: int,
        num_failed_prun_nets: np.ndarray,
        sparsities,
        old_run_chp: np.ndarray,
        box_width: float = 2  # default box width
    ):
        failed_prun_plt = ax.bar(
            old_run_chp, 
            num_failed_prun_nets/runs*100, 
            color='tab:red', 
            alpha=0.3, 
            width=box_width,
            label='Failed networks',
        )
        sparsity_plt = ax.plot(
            old_run_chp,
            sparsities,
            label='Sparsity'
        )
        ax.set_ylim([0, 100])
        ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1.02, 1.15), frameon=False)
        return failed_prun_plt, sparsity_plt


    def plotPrunMetrics(
        self, 
        method_config: PrunConfig | MethodPrunConfig,
        unpruned_metrics: pd.DataFrame, 
        prun_metrics: pd.DataFrame,
        prun_sparsities: pd.DataFrame,
    ) -> MetricFigures:
        """Compares the pruned with the unpruned metrics of every model. 

        INPUTS
        ------ 
            ``unpruned_metrics`` are the loaded train and test metrics in dataframe for every model. 

            ``prun_metrics`` are the pruned train and test metrics in dataframe for every model. 

        RETURN
        ------
            ``figures`` is a list of figure with a metrics comparison betweem pruned 
            and unpruned. 
        """
        old_changed_hyperparams = prun_metrics.index.get_level_values(level=1)
        new_changed_hyperparams = prun_metrics.index.get_level_values(level=0)
        runs = prun_metrics.index.get_level_values(level='run').nunique()
        old_run_chp = old_changed_hyperparams.to_list()[0::runs]

        subplot_width = 2 
        multipl_cb = MULTIPLIER_WIDTH_COLORBAR

        figures = MetricFigures([])
        for metric_name in self.selected_metric_names:
            env_name = ENVIROMENT_NAMES[1]
            unprun_metric = unpruned_metrics.xs(metric_name, level=1, axis=1)[env_name]
            prun_metric = prun_metrics.xs(metric_name, level=1, axis=1)[env_name]
            np_unprun_metric = unprun_metric.to_numpy()
            np_prun_metric = prun_metric.to_numpy()

            if metric_name=='R²':
                # lims
                method_str = get_method_str(method_config)
                lims_plot1 = get_thesis_plot_lims3d(method_str)

                # axis labels
                first_plot_xlabel = r'$\mathrm{R}^{2}_{\mathrm{Unpr}}$'
                first_plot_ylabel = r'$\mathrm{R}^{2}_{\mathrm{Pr}}$'
                second_plot_ylabel = r'$\Delta \mathrm{R}^2$'# r'$\mathrm{R}^{2}_{\mathrm{Pr}} - \mathrm{R}^{2}_{\mathrm{Unpr}}$'
                diff = np_prun_metric - np_unprun_metric
                

            elif metric_name=='RMSE':
                # lims
                lims_plot1 = get_plot_lim(
                    np.concatenate((np_unprun_metric, np_prun_metric)),
                    scale='log'
                )
                ylims_plot2 = get_plot_lim(diff)

                # axis labels
                multipl_cb *= 1.1
                first_plot_xlabel = r'$\mathrm{RMSE}_{\mathrm{Unpr}}$'
                first_plot_ylabel = r'$\mathrm{RMSE}_{\mathrm{Pr}}$'
                second_plot_ylabel = r'$\mathrm{RMSE}_{\mathrm{Pr}} / \mathrm{RMSE}_{\mathrm{Unpr}}$'
                diff = np_prun_metric / np_unprun_metric

            mean_diff = diff.mean()
            xlims_plot2 = get_plot_lim(np.array(old_run_chp)) 
            metric_diff = pd.DataFrame(diff, index=prun_metric.index)
                        

            ## FIGURE
            fig = plt.figure(
                figsize=set_size((PLOT_HEIGHT*subplot_width*multipl_cb, PLOT_HEIGHT)), 
                constrained_layout=True,
            )
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[2, 3])
            
            ## FIRST PLOT
            ax1 = fig.add_subplot(gs[:, 0],)
            ax1.grid(which='major', linewidth=0.5)
            ax1.grid(which='minor', linestyle=':', linewidth=0.2)
            ax1.set_xlabel(first_plot_xlabel)
            ax1.set_ylabel(first_plot_ylabel)
            set_equal_lims(ax1, lims_plot1)
            pl1 = self.firstPlot(
                ax1,
                unprun_metric,
                prun_metric,
                old_run_chp
            )

            if lims_plot1[0] < 0. and lims_plot1[1] > 1.:
                ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

            # set scale to log if RMSE
            if metric_name == 'RMSE':
                ax1.set_xscale('log')
                ax1.set_yscale('log')

            default_colorbar(fig, ax1, pl1, r'size [${}$]'.format(new_changed_hyperparams.name), ticks=old_run_chp)


            # find failed networks and survived ones of pruned and unpruned
            num_failed_unprun_nets, num_failed_prun_nets, survived_nets = find_failed_nets(
                unprun_metric, prun_metric, metric_diff
            )
            box_width = old_run_chp[1]/old_run_chp[0] if len(old_run_chp) < 0 else 0.8
            if metric_name == 'R²':
                surv_nets_conc = np.concatenate(survived_nets)
                ylims_plot2 = get_plot_lim(surv_nets_conc)
                ylims_plot2, legend_loc = get_legend_loc(ylims_plot2, survived_nets)


            # SECOND TOP PLOT
            ax2_top: plt.Axes = fig.add_subplot(gs[0, 1]) if metric_name == 'R²' else fig.add_subplot(gs[:, 1])
            plot_y_zero_line(ax2_top, xlims_plot2)     
            if ylims_plot2[0] < mean_diff < ylims_plot2[1]:
                plot_avg_r2(ax2_top, xlims_plot2, mean_diff)
            pl2 = self.secondPlot_top(
                ax2_top,
                survived_nets,
                old_run_chp,
                colors=pl1.to_rgba(range(len(old_run_chp))),
                box_width=box_width*2
            )

            ax2_top.set_ylabel(second_plot_ylabel)
            ax2_top.set_ylim(ylims_plot2)
            
            if metric_name == 'R²' and ylims_plot2[0] < mean_diff < ylims_plot2[1]:
                make_legend_r2_diff(ax2_top, legend_loc)
            elif metric_name != 'R²' and ylims_plot2[0] < mean_diff < ylims_plot2[1]:
                ax2_top.legend(loc='best', frameon=False)
            


            # SECOND BOTTOM PLOT
            if metric_name == 'R²':
                ax2_top.xaxis.set_tick_params(labelbottom=False)
                ax2_bot: plt.Axes = fig.add_subplot(gs[1, 1], sharex=ax2_top)
                ax2_bot.set_ylabel(r'Percent')
                self.secondPlot_bottom(
                    ax2_bot,
                    runs,
                    num_failed_prun_nets,
                    prun_sparsities['total_sparsities'].groupby(level='old_nodes').mean(),
                    old_run_chp,
                    box_width,
                )
                ax2_bot.set_xlabel(r'size [${}$]'.format(new_changed_hyperparams.name))
                ax2_bot.xaxis.set_major_locator(ticker.MaxNLocator(10))
                ax2_bot.yaxis.set_major_locator(ticker.MaxNLocator(4))
                set_correct_latex_xticks(ax2_bot)
                ax2_bot.set_xlim(xlims_plot2)
            else:
                ax2_top.set_xlabel(r'size [${}$]'.format(new_changed_hyperparams.name))
                ax2_top.xaxis.set_major_locator(ticker.MaxNLocator(10))
                ax2_top.yaxis.set_major_locator(ticker.MaxNLocator(4))
                set_correct_latex_xticks(ax2_top)
                ax2_top.set_xlim(xlims_plot2)

                
                
            plt.close()
            figures.append(fig)
        return figures


    def plotIterSparseMetrics(
        self,
        mean_metrics: pd.DataFrame,
        sparsity: pd.DataFrame,
    ) -> IterPrunFigures:
        figures = IterPrunFigures([])
        for (_, net_mean_metrics), (_, net_sparsity) in zip(mean_metrics.groupby(level=0), sparsity.groupby(level=0)):
            net_sparsity_series = net_sparsity.iloc[:, 0]
            x_level_index = 1
            fig = self.plotMeanMetrics(net_mean_metrics, x_level_index=x_level_index, sparsity_plot=True)
            top_ax = fig.axes[0] if isinstance(fig.axes, list) else fig.axes
            set_top_xticks(top_ax, net_sparsity_series.to_list(), net_sparsity_series.name)
            
            figures.append(fig)
        return figures


    @abstractmethod
    def plotModelData(self):
        pass


    @abstractmethod
    def plotOrigFun(self):
        pass