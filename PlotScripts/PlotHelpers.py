import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import matplotlib.cm as cm

from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib.axis import Axis
from matplotlib.colorbar import Colorbar
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.legend import Legend




# =============================================================================
# COLORBAR
# =============================================================================
def default_colorbar(
    fig: Figure, 
    ax: plt.Axes, 
    plot: PatchCollection, 
    bar_label: str,
    ticks: list = None
) -> Colorbar:
    """Sets a colorbar to the right of the given ax. 

    INPUTS
    ------
        ``fig`` is the figure. 

        ``ax`` is the ax of the figure. 

        ``plot`` is the plot as PathCollection (e.g. contour).

        ``bar_label`` is the label on the right side of the cb.

        ``ticks`` are the ticks on the right side of the cb. 
        Neede if previously the. 

    RETURN
    ------
        ``cbar`` is the colorbar instance.
    """
    
    cbar = fig.colorbar(plot, ax=ax, label=bar_label)
    if ticks is not None:
        cbar.ax.set_yticks(np.arange(0, len(ticks)))
        cbar.ax.set_yticklabels(ticks)
        if len(ticks) > 10:
            cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        if len(ticks) <= 20: 
            cbar.ax.tick_params(axis='both', length=0)

    return cbar
        

def default_two_axis_colorbar(
    fig: Figure, 
    ax: plt.Axes, 
    plot: PatchCollection, 
    bar_label_right: str,
    ticks_right: list,
    bar_label_left: str,
    ticks_left: list
) -> Colorbar:
    """Sets a colorbar with two axes to the right of the given ax. 

    INPUTS
    ------
        ``fig`` is the figure. 

        ``ax`` is the ax of the figure. 

        ``plot`` is the plot as PathCollection (e.g. contour).

        ``ticks_right`` are the ticks on the right side of the cb. 

        ``bar_label_right`` is the label on the right side of the cb.

        ``ticks_left`` are the ticks on the left side of the cb.  

        ``bar_label_left`` is the label on the left side of the cb.  

    RETURN
    ------
        ``cbar`` is the colorbar instance.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%')
    cbar = fig.colorbar(plot, cax=cax)

    ax_right = cbar.ax
    ax_left = ax_right.twinx()

    tickz = np.arange(0, len(ticks_right))
        
    ax_right.set_yticks(tickz)
    ax_right.set_yticklabels(ticks_right)
    ax_right.set_ylabel(bar_label_right)
    ax_right.yaxis.set_ticks_position('right')
    ax_right.yaxis.set_label_position('right')
    if len(ticks_right) <= 20: 
        ax_right.tick_params(axis='both', length=0)

    ax_left.set_yticks(tickz)
    ax_left.set_ylim(ax_right.get_ylim())
    ax_left.set_yticklabels(
        ticks_left if not isinstance(ticks_left[0], float) else make_float_labels(ticks_left)
    )
    ax_left.set_ylabel(bar_label_left)
    ax_left.yaxis.set_ticks_position('left')
    ax_left.yaxis.set_label_position('left')
    if len(ticks_left) <= 20: 
        ax_left.tick_params(axis='both', length=0)
    return cbar


def make_float_labels(labels):
    return [f'{label:.1f}' for label in labels]





# =============================================================================
# CMAP
# =============================================================================
def chooseCmap(num_values: int):
    if num_values <= 10: 
        cmap = cm.get_cmap('tab10')
        return ListedColormap(cmap(range(num_values))) 
    elif num_values > 10 and num_values <= 20:
        cmap = cm.get_cmap('tab20') 
        return ListedColormap(cmap(range(num_values)))
    else: 
        return cm.get_cmap('brg')





# =============================================================================
# LIMS
# =============================================================================
def set_equal_lims(ax: plt.Axes, lims: list):
    ax.set_aspect('equal')
    if lims[0]!=lims[1]:
        ax.set_ylim(lims) 
        ax.set_xlim(lims) 
        ax.plot(lims, lims, 'tab:gray', linewidth=0.5)
        
        
def get_plot_lim(values: np.ndarray, set_min = None, set_max = None, scale = 'lin'):
    min_xy = values.min() if set_min is None else set_min
    max_xy = values.max() if set_max is None else set_max
    if scale == 'lin':
        min_plot = min_xy - (max_xy-min_xy)*0.05
        max_plot = max_xy + (max_xy-min_xy)*0.05
    elif scale == 'log':
        log_diff = 10**(np.log10(max_xy / min_xy) * 0.05)
        min_plot = min_xy / log_diff
        max_plot = max_xy * log_diff
    else:
        raise(ValueError('Not the correct scale string!', scale))
    return [min_plot, max_plot]


def plot_y_zero_line(ax: plt.Axes, xlims):
    y = [0 for _ in range(len(xlims))]
    ax.plot(xlims, y, color='tab:gray', linestyle='-', linewidth=1.2)
    
    
def plot_avg_r2(ax: plt.Axes, xlims, mean_metric):
    y = [mean_metric for _ in range(len(xlims))]
    ax.plot(xlims, y, color='tab:blue', linestyle='--', linewidth=1.2, label=r'$\Delta \mathrm{R}^2_{\mathrm{Avg}}$')
    

def get_thesis_plot_lims3d(method_str: str):
    if 'syn_flow_5_' in method_str:
        return get_plot_lim(None, set_min=0.69, set_max=0.93) 
    elif 'lottery_ticket_' in method_str and '_str_' in method_str and '_np_' not in method_str and '_lnp_' not in method_str:
        return get_plot_lim(None, set_min=0.69, set_max=0.95)
    elif 'lottery_ticket_' in method_str and '_str_' in method_str and '_lnp_' in method_str:
        return get_plot_lim(None, set_min=0.66, set_max=0.93)
    else:
        return get_plot_lim(None, set_min=0., set_max=1.)
    

def get_thesis_plot_lims2d(method_str: str):
    if 'syn_flow_5_' in method_str:
        return get_plot_lim(None, set_min=0.8, set_max=1.)
    elif 'lottery_ticket_' in method_str and '_str_' in method_str and '_np_' not in method_str and '_lnp_' not in method_str:
        return get_plot_lim(None, set_min=0.8, set_max=1.)
    elif 'lottery_ticket_' in method_str and '_str_' in method_str and '_lnp_' in method_str:
        return get_plot_lim(None, set_min=0.76, set_max=1.)
    else:
        return get_plot_lim(None, set_min=0., set_max=1.)





# =============================================================================
# TICKS
# =============================================================================
def make_less_ticks(ax: plt.Axes, every_nth: int):
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
            

def set_correct_latex_xticks(ax: plt.Axes):
    xticks = ax.get_xticks()
    new_xtick_labels = [str(int(tick)) for tick in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(new_xtick_labels)


def set_yticks_int_only(ax: plt.Axes):
    ticks = ax.get_yticks()
    min_tick = math.ceil(min(ticks))
    max_tick = math.floor(max(ticks))
    diff = max_tick - min_tick + 1
    if not (isinstance(ticks[0], np.integer) and len(ticks) <= diff):
        new_ticks = range(min_tick, max_tick + 1)
        ax.set_yticks(new_ticks)


def set_xticks_int_only(ax: plt.Axes):
    lims = ax.get_xlim()
    ticks = ax.get_xticks()
    min_tick = math.ceil(min(ticks))
    max_tick = math.floor(max(ticks))
    if not isinstance(ticks[0], np.integer) and ticks[1]-ticks[0]<=1:
        new_ticks = range(min_tick, max_tick + 1)
        ax.set_xticks(new_ticks)
        ax.set_xlim(lims)


def set_top_xticks(ax: plt.Axes, tick_labels: list, xlabel: str):
    ticks = ax.get_xticks()
    lims = ax.get_xlim()

    top_ax = ax.twiny()
    top_ax.set_xlabel(xlabel)
    top_ax.set_xticks(ticks)
    top_ax.set_xlim(lims)

    if isinstance(tick_labels[0], float):
        tick_labels = make_float_labels(tick_labels)

    for i in range(len(ticks)):
        if ticks[i] < lims[0]:
            tick_labels.insert(0, '0')
        elif ticks[i] > lims[1]:
            tick_labels.append('0')
            
    top_ax.set_xticklabels(tick_labels)
    top_ax.xaxis.set_ticks_position('top')
    top_ax.xaxis.set_label_position('top')
    
    if len(ticks) > 10:
        top_ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    return top_ax


def use_ScalarFormater(axis: Axis):
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-2, 2))  # Change the range of exponents that are shown
    axis.set_major_formatter(formatter)





# =============================================================================
# LEGEND STUFF
# =============================================================================
def adjust_fig_size_legend_above(fig: Figure, legend: Legend):
    legend_height = legend.get_window_extent().height / fig.get_dpi()
    fig.set_figheight(fig.get_figheight() + legend_height)
    
    
def get_legend_str(method_str: str):
    orig_method_str = method_str
    if 'magnitude' in method_str:
        legend_str = r'Magnitude'
        method_str = method_str.replace('magnitude_', '')
    elif 'lottery_ticket' in method_str:
        legend_str = r'LTH'
        method_str = method_str.replace('lottery_ticket_', '')
    elif 'finetune' in method_str:
        legend_str = r'Finetune'
        method_str = method_str.replace('finetune_', '')
    elif 'syn_flow' in method_str:
        legend_str = r'SynFlow'
        method_str = method_str.replace('syn_flow_', '')
        
    if '_l1_' in method_str or '_global_' in method_str:
        legend_str += r' $L^1$'
    elif '_l2_' in method_str:
        legend_str += r' $L^2$'
    elif '_linf_' in method_str:
        legend_str += r' $L^{\\inf}$'
        
    if method_str.startswith('0'):
        legend_str += r' single-shot'
    else:
        legend_str += r' {} iterations'.format(find_first_number_in_str(method_str))
        if 'exp_' in method_str and '_unstr_' in method_str:
            legend_str += r' exp'
      
    if '_np_' in method_str:
        legend_str += r' with npp'
    if '_lnp_' in method_str:
        legend_str += r' with linpp'
    return legend_str


def find_first_number_in_str(str_with_number: str, end_char: str = ''):
    for i, str_char in enumerate(str_with_number):
        if str_char == end_char: 
            return ''
        if str_char.isdigit():
            return str_char + find_first_number_in_str(str_with_number[i+1:], end_char='_')
        
    
def make_legend_r2_diff(ax: plt.Axes, loc: str):
    if loc=='bottom left':
        ax.legend(loc='lower left', bbox_to_anchor=(-0.02, -0.13), frameon=False)
    elif loc=='bottom right':
        ax.legend(loc='lower right', bbox_to_anchor=(1.02, -0.13), frameon=False)
    elif loc=='top left':
        ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.13), frameon=False)
    elif loc=='top right':
        ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1.13), frameon=False)
        

def get_legend_loc(ylims: list, data: list[np.ndarray]):
    def _furthest_diff(ylims: list, data: list[np.ndarray]):
        data_conc = np.concatenate(data)
        if data_conc.size != 0:
            bot_lim = data_conc.min()
            top_lim = data_conc.max()
            bottom_diff = bot_lim - ylims[0]
            top_diff = ylims[1] - top_lim
        else:
            bot_lim, top_lim, bottom_diff, top_diff = (0, 0, 0, 0) 
        return bot_lim, top_lim, bottom_diff, top_diff, 
    
    
    data_len = len(data)
    bottom_left_lim, top_left_lim, bottom_left_diff, top_left_diff = _furthest_diff(ylims, data[:int(data_len/3)])
    bottom_right_lim, top_right_lim, bottom_right_diff, top_right_diff = _furthest_diff(ylims, data[int(data_len*2/3):])
    
    must_lims = [top_left_lim, top_right_lim, bottom_right_lim, bottom_left_lim] 
    diffs = [top_left_diff, top_right_diff, bottom_right_diff, bottom_left_diff]
    locs = ['top left' , 'top right', 'bottom right', 'bottom left']
    
    diff_idx = diffs.index(max(diffs))
    legend_loc = locs[diff_idx]
    idx = 0 if 'bottom' in legend_loc else 1 
    
    updated_lims = get_updated_lims_for_legend(ylims, must_lims[diff_idx], idx, 0.5)
    return updated_lims, legend_loc


def get_updated_lims_for_legend(lims: list, must_lim: float, idx: int, percent: float):
    if idx==0:
        must_lim = must_lim - (lims[1] - must_lim)*percent
        if must_lim < lims[0]:
            lims[0] = must_lim
    elif idx==1:
        must_lim = must_lim + (must_lim - lims[0])*percent
        if must_lim > lims[1]:
            lims[1] = must_lim
    return lims
    
        



# =============================================================================
# STYLE
# =============================================================================
def get_linestyle(method_str: str, counter: dict):
    MAG_COLORS = ('tab:blue', 'tab:cyan', 'steelblue', 'dodgerblue', 'mediumblue', 'cornflowerblue')
    LTH_COLORS = ('tab:red', 'tab:orange', 'tab:brown', 'maroon', 'orangered', 'coral', 'rosybrown', 'chocolate', 'indianred')
    FT_COLORS = ('olive', 'tab:green', 'mediumseagreen', 'darkolivegreen', 'yellow', 'green', 'mediumturquoise', 'lightseagreen', 'springgreen', 'darkseagrean')
    SF_COLORS = ('tab:purple', 'tab:pink')
    STYLES = ('-', '-', '--', ':', '-.', (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)), (0, (5, 1)), (0, (10, 5, 1, 5)))
    if 'magnitude' in method_str:
        count = counter['mag']
        color = MAG_COLORS[count]
        style = STYLES[counter['mag']]
        counter['mag'] += 1
    elif 'lottery_ticket' in method_str:
        count = counter['lth']
        color = LTH_COLORS[count]
        style = STYLES[counter['lth']]
        counter['lth'] += 1
    elif 'finetune' in method_str:
        count = counter['ft']
        color = FT_COLORS[count]
        style = STYLES[counter['ft']]
        counter['ft'] += 1
    elif 'syn_flow' in method_str:
        count = counter['sf']
        color = SF_COLORS[count]
        style = STYLES[counter['sf']]
        counter['sf'] += 1
    return style, color





# =============================================================================
# CALCULATIONS
# =============================================================================
def get_random_runs_metrics(
        unprun_metrics: pd.Series, 
        prun_metrics: pd.Series, 
        size: int
    ):
        unprun_runs = unprun_metrics.index.get_level_values(level='run')
        prun_runs = prun_metrics.index.get_level_values(level='run')
        if (unprun_runs == prun_runs).all():
            mask = np.isin(
                unprun_runs,
                np.random.choice(unprun_runs.unique(), size=size, replace=False)
            )
            return unprun_metrics.loc[mask], prun_metrics.loc[mask] 
        else:
            raise ValueError('Metric Dataframes have not the same runs!')
        

def find_failed_nets(env_unprun_metric: pd.Series, env_prun_metric: pd.Series, env_metric_diff: pd.Series):
    unprun_arr = env_unprun_metric.unstack(level=0).to_numpy()
    prun_arr = env_prun_metric.reset_index(level=0, drop=True).unstack(level=0).to_numpy()
    diff_arr = env_metric_diff.reset_index(level=0, drop=True).unstack(level=0).to_numpy()

    net_failed_unprun = unprun_arr < 0
    num_failed_unprun_nets = np.count_nonzero(net_failed_unprun, axis=0)

    net_failed_prun = prun_arr < 0
    num_failed_prun_nets = np.count_nonzero(net_failed_prun, axis=0)
    net_survived = ~net_failed_prun
    survived_nets = [diff_arr[net_survived[:, i], i] for i in range(diff_arr.shape[1])]
    
    net_failed_both = np.logical_and(net_failed_unprun, net_failed_prun)
    num_failed_both_nets = np.count_nonzero(net_failed_both, axis=0)
    
    num_failed_unprun_nets -= num_failed_both_nets
    num_failed_prun_nets -= num_failed_both_nets
    
    return num_failed_unprun_nets, num_failed_prun_nets, survived_nets


def get_unique_level_values(frame: pd.DataFrame):
    idxs_without_run = frame.index.droplevel('run')
    num_idx_levels = idxs_without_run.nlevels
    methods = []
    for i in range(num_idx_levels):
        methods.append(frame.index.get_level_values(i))
    return pd.MultiIndex.from_arrays(methods).unique()


def set_size(fig_w_h: tuple[int], width_pt = 483.69688, fraction = 1, flip = False):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
        ``fig_w_h`` is the size of 

        ``width_pt`` 

    Returns
    -------
        ``fig_dim``
    """
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if not flip:
        fig_width_in = width_pt * fraction * inches_per_pt
        fig_height_in = fig_width_in * fig_w_h[1] / fig_w_h[0]
    else:
        fig_height_in = width_pt * fraction * inches_per_pt
        fig_width_in = fig_height_in * fig_w_h[0] / fig_w_h[1]

    return (fig_width_in, fig_height_in)





# =============================================================================
# LATEX FONTS
# =============================================================================
def make_latex_fonts():
    #plt.style.use('seaborn')
    plt.rcParams.update({
        'font.family': "serif",  # use serif/main font for text elements
        'font.size': 11,         # use font size of 11pt
        'text.usetex': True,     # use inline math for ticks
        'pgf.rcfonts': False     # don't setup fonts from rc parameters
        })