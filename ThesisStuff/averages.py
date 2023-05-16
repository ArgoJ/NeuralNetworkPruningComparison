import pandas as pd
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from changeStuff import on_every_Prune_config_do, on_every_base_config_do
from Saves import save_JSON, Saves
from Loads import Loadings
from Constants import METRIC_NAMES

from ThesisStuff.prunMethodCompPerf import get_performance_frames, make_method_str_from_tuple





def averageMedianMetricDiffs(base_path: Path, prune_path: Path):
    if os.path.basename(base_path)=='02_16_20_25':
        load_cls = Loadings(base_path)
        prun_base_path = os.path.dirname(os.path.dirname(prune_path))
        load_cls.updateDirectorys(prun_base_path)
        prun_folder = os.path.basename(prune_path)
        load_cls.setPrunMethodDirectorys(prun_folder)

        orig_metrics, _ = load_cls.loadMetrics()
        prun_metrics, _ = load_cls.loadPrunMetrics()

        orig_metrics = orig_metrics.xs(key=METRIC_NAMES[0], axis=1, level=1, drop_level=False)
        prun_metrics = prun_metrics.xs(key=METRIC_NAMES[0], axis=1, level=1, drop_level=False)
        prun_metrics = prun_metrics.droplevel(level='nodes')
        prun_metrics.index = prun_metrics.index.set_names(['nodes', 'run'])

        prun_metrics_surv = prun_metrics[prun_metrics > 0]
        diff_metrics = prun_metrics_surv - orig_metrics
        
        diff_metrics_medians = diff_metrics.groupby('nodes', as_index=False).apply(lambda x: x.median(axis=0)) # * 100 - 100
        average_medians = diff_metrics_medians.mean()
        
        avr_medians_test = average_medians.loc[('test', 'R²')]
        max_median_test = diff_metrics_medians.max().loc[('test', 'R²')]
        print(f'max median: {max_median_test}')
        print(f'avr. median: {avr_medians_test}')
        
        return avr_medians_test



def metric_difference(prun_metrics: pd.DataFrame, orig_metrics: pd.DataFrame):
    diff = prun_metrics - orig_metrics
    diff_means = diff.groupby('nodes', group_keys=False).apply(lambda x : x.mean(axis=0))
    return 'Difference', diff_means

def metric_percent(prun_metrics: pd.DataFrame, orig_metrics: pd.DataFrame):
    diff = prun_metrics / orig_metrics * 100 - 100
    diff_means = diff.groupby('nodes', group_keys=False).apply(lambda x : x.mean(axis=0))
    return 'Percent', diff_means
    
def metric_prun(prun_metrics: pd.DataFrame, orig_metrics: pd.DataFrame):
    diff = prun_metrics
    diff_means = diff.groupby('nodes', group_keys=False).apply(lambda x : x.mean(axis=0))
    return 'R²', diff_means

def averageMetricDiffs(base_path: Path, prune_path: Path, diff_function=metric_difference, start_nodes: int = None, end_nodes: int = None):
    if os.path.basename(base_path)=='02_16_20_25' or os.path.basename(base_path)=='02_16_14_31':
        load_cls = Loadings(base_path)
        prun_base_path = os.path.dirname(os.path.dirname(prune_path))
        load_cls.updateDirectorys(prun_base_path)
        prun_folder = os.path.basename(prune_path)
        load_cls.setPrunMethodDirectorys(prun_folder)

        orig_metrics, orig_mean_metrics = load_cls.loadMetrics()
        prun_metrics, prun_mean_metrics = load_cls.loadPrunMetrics()

        orig_metrics = orig_metrics.xs(key=METRIC_NAMES[0], axis=1, level=1, drop_level=False)
        prun_metrics = prun_metrics.xs(key=METRIC_NAMES[0], axis=1, level=1, drop_level=False)
        prun_metrics = prun_metrics.droplevel(level='nodes')
        prun_metrics.index = prun_metrics.index.set_names(['nodes', 'run'])

        print_name, diff_metrics_impr = diff_function(prun_metrics, orig_metrics)
        diff_metrics_impr = diff_metrics_impr.loc[slice(start_nodes, end_nodes)]
        
        max_acc = diff_metrics_impr.max(axis=0).loc[('test', 'R²')]
        min_acc = diff_metrics_impr.min(axis=0).loc[('test', 'R²')]
        mean_acc = diff_metrics_impr.mean(axis=0).loc[('test', 'R²')]
        print(f'Max {print_name}: {max_acc:.3f}')
        print(f'Min {print_name}: {min_acc:.3f}')
        print(f'Mean {print_name}: {mean_acc:.3f}')

        return mean_acc
    
    
    
def averageMetricOrig(dim: int, base_path: Path, start_nodes: int = None, end_nodes: int = None):
    time_folder = os.path.basename(base_path)
    if time_folder=='02_16_20_25' or time_folder=='02_16_14_31':
        load_cls = Loadings(base_path)

        orig_metrics, orig_mean_metrics = load_cls.loadMetrics()
        orig_metrics = orig_metrics.xs(key=METRIC_NAMES[0], axis=1, level=1, drop_level=False)
        orig_metrics = orig_metrics.xs(key=slice(start_nodes, end_nodes), level='nodes')
        
        max_acc = orig_metrics.max(axis=0).loc[('test', 'R²')]
        min_acc = orig_metrics.min(axis=0).loc[('test', 'R²')]
        mean_acc = orig_metrics.mean(axis=0).loc[('test', 'R²')]
        print(f'DIM: {dim}, FOLDER: {time_folder}, ORIGINAL METRICS')
        print(f'Max R²: {max_acc:.3f}')
        print(f'Min R²: {min_acc:.3f}')
        print(f'Mean R²: {mean_acc:.3f}')

        return mean_acc





def averagePerfDiffs(
    dim: int,
    base_path: str,
    perf_name: str = '',
    fun2apply_2ALL = None,
    fun2apply_1NET = None,
):
    if os.path.basename(base_path)=='02_16_14_31' or os.path.basename(base_path)=='02_16_20_25':
        orig_perfs, prun_perfs = get_performance_frames(base_path)
        orig_perf: pd.DataFrame = orig_perfs[0].loc[:, perf_name]
        
        print(f'Unpruned: {orig_perf.mean():.3f}')
        
        for (method_tup, prun_perf), (method_tup_mean, prun_mean_perf) in zip(
            prun_perfs[0].groupby(level=list(range(7))), 
            prun_perfs[1].groupby(level=list(range(7)))
        ):
            method_str = make_method_str_from_tuple(method_tup)
            prun_perf.index = prun_perf.index.droplevel(list(range(7)))
            prun_perf: pd.DataFrame = prun_perf.loc[:, perf_name]
            
            df_mean, df_min, df_max = fun2apply_2ALL(prun_perf, orig_perf)
            print(f'DIM: {dim}, ARCH: {os.path.basename(base_path)}, METHOD: {method_str}:')
            print(f'MEAN: {df_mean:.1f}')
            print(f'MIN: {df_min:.1f}')
            print(f'MAX: {df_max:.1f}')
            
            key = 120
            net_mean = fun2apply_1NET(prun_perf, orig_perf, key=key)
            print(f'{key} nodes: {net_mean:.1f}')
    return None


# FOR ALL
def df_all_diff(prun_df: pd.DataFrame, orig_df: pd.DataFrame):
    df_mean = (prun_df - orig_df).mean()
    df_min = (prun_df - orig_df).min()
    df_max = (prun_df - orig_df).max()
    return df_mean, df_min, df_max

def df_all_perc(prun_df: pd.DataFrame, orig_df: pd.DataFrame):
    df_mean = ((prun_df / orig_df).mean() - 1) * 100
    df_min = ((prun_df / orig_df).min() - 1) * 100
    df_max = ((prun_df / orig_df).max() - 1) * 100
    return df_mean, df_min, df_max

def df_all_compr(prun_df: pd.DataFrame, orig_df: pd.DataFrame):
    df_mean = (orig_df / prun_df).mean()
    df_min = (orig_df / prun_df).min()
    df_max = (orig_df / prun_df).max()
    compr = orig_df / prun_df
    fig = plt.figure()
    compr.groupby(level='nodes').mean().plot(x='nodes')
    compr.groupby(level='nodes').max().plot(x='nodes')
    compr.groupby(level='nodes').min().plot(x='nodes')
    return df_mean, df_min, df_max



# FOR NET
def df_net_diff(prun_df: pd.DataFrame, orig_df: pd.DataFrame, key):
    return prun_df.xs(key, level='nodes').mean() - orig_df.xs(key, level='nodes').mean()

def df_net_perc(prun_df: pd.DataFrame, orig_df: pd.DataFrame, key):
    return (prun_df.xs(key, level='nodes').mean() / orig_df.xs(key, level='nodes').mean() - 1 ) * 100

def df_net_compr(prun_df: pd.DataFrame, orig_df: pd.DataFrame, key):
     return orig_df.xs(key, level='nodes').mean() / prun_df.xs(key, level='nodes').mean()
 
 




if __name__ == '__main__':
    # start_nodes=None
    # end_nodes=None
    # on_every_base_config_do(averageMetricOrig, start_nodes=start_nodes, end_nodes=end_nodes)
    # outputs = on_every_Prune_config_do(averageMetricDiffs, diff_function=metric_prun, start_nodes=start_nodes, end_nodes=end_nodes)
    
    
    on_every_base_config_do(
        averagePerfDiffs, 
        perf_name='time [us]', 
        fun2apply_2ALL=df_all_perc,
        fun2apply_1NET=df_net_perc
    ) 
    plt.show()
    
    # 'FLOPs', 'size [bytes]', 'time [us]', 'R²'