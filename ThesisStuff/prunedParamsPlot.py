import pandas as pd
import matplotlib.pyplot as plt
import os, sys


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from PlotScripts.PlotHelpers import *
from DataClassesJSON import ConfigData
from Loads import loadConfig





def get_hidden_nodes(config: ConfigData):
    start_hidden_nodes = config.nodes*config.layer
    step_hidden_nodes = config.node_step*config.layer
    end_hidden_nodes = start_hidden_nodes + step_hidden_nodes*config.networks
    hidden_nodes = [nodes for nodes in range(start_hidden_nodes, end_hidden_nodes, step_hidden_nodes)]
    return np.array(hidden_nodes)


def plot_local_unstr_pruned_params(config: ConfigData, amount = 0.2, lims = [-0.02, 0.3]):
    hidden_nodes = get_hidden_nodes(config)

    oneL_end_h_nodes = config.nodes + config.node_step*config.networks
    params = [
        [nodes**2 if l!=0 else nodes*config.inputs for l in range(config.layer)] 
        for nodes in range(config.nodes, oneL_end_h_nodes, config.node_step)
    ]
    for idx, nodes in enumerate(range(config.nodes, oneL_end_h_nodes, config.node_step)):
        params[idx].append(nodes)
    

    params_np = np.array(params)
    pruned_p = np.round(params_np*amount)
    percent_pruned_p = pruned_p / params_np

    total_params = params_np.sum(axis=1)
    total_prun_p = pruned_p.sum(axis=1)
    percent_total_prun_p = total_prun_p / total_params
    

    # figure
    fig = plt.figure(figsize=set_size((9,5), fraction=0.7), constrained_layout=True)
    plt.plot(hidden_nodes, percent_pruned_p[:, [0, 1, -1]], label=('first hidden layer', 'other hidden layers', 'output layer'))
    plt.plot(hidden_nodes, percent_total_prun_p, label='total pruned parameters')
    plt.legend()
    plt.ylim(lims)
    return fig


def plot_local_str_pruned_params(config: ConfigData, amount = 0.2, lims = [-0.02, 0.3]):
    hidden_nodes = get_hidden_nodes(config)

    end_nodes = config.nodes + config.node_step*(config.networks-1)
    params = np.linspace(config.nodes, end_nodes, config.networks)
    pruned_p = np.round(params*amount)
    percent_pruned_p = pruned_p / params


    # calc per total percentage pruned
    oneL_end_h_nodes = config.nodes + config.node_step*config.networks
    params_perL = [
        [nodes**2 if l!=0 else nodes*config.inputs for l in range(config.layer)] 
        for nodes in range(config.nodes, oneL_end_h_nodes, config.node_step)
    ]
    pruned_p_perL = [
        [pruned_nodes*nodes if l!=0 else pruned_nodes*config.inputs for l in range(config.layer)] 
        for pruned_nodes, nodes in zip(pruned_p, range(config.nodes, oneL_end_h_nodes, config.node_step))
    ]
    for idx, nodes in enumerate(range(config.nodes, oneL_end_h_nodes, config.node_step)):
        params_perL[idx].append(nodes)
        pruned_p_perL[idx].append(0.)

    params_perL_np = np.array(params_perL)
    total_params = params_perL_np.sum(axis=1)
    pruned_p_perL_np = np.array(pruned_p_perL)
    total_pruned_p = pruned_p_perL_np.sum(axis=1)
    percent_total_prun_p = total_pruned_p / total_params
    

    # figure
    fig = plt.figure(figsize=set_size((9,5), fraction=0.7), constrained_layout=True)
    plt.plot(hidden_nodes, percent_pruned_p, label='pruned nodes')
    plt.plot(hidden_nodes, percent_total_prun_p, label='total pruned parameters')
    plt.legend()
    plt.ylim(lims)
    return fig


def plot_global_unstr_pruned_params(config: ConfigData, amount = 0.2, lims = [-0.02, 0.3]):
    hidden_nodes = get_hidden_nodes(config)

    oneL_end_h_nodes = config.nodes + config.node_step*config.networks
    params = [
        [nodes**2 if l!=0 else nodes*config.inputs for l in range(config.layer)] 
        for nodes in range(config.nodes, oneL_end_h_nodes, config.node_step)
    ]
    for idx, nodes in enumerate(range(config.nodes, oneL_end_h_nodes, config.node_step)):
        params[idx].append(nodes)

    total_params = np.array(params).sum(axis=1)
    total_pruned_p = np.round(total_params*amount)
    percent_total_prun_p = total_pruned_p / total_params


    # figure
    fig = plt.figure(figsize=set_size((9,5), fraction=0.7), constrained_layout=True)
    plt.plot(hidden_nodes, percent_total_prun_p)
    plt.ylim(lims)
    return fig




if __name__ == '__main__':
    config_path = input('type in the path of the config json.\n').replace('"', '')
    config = loadConfig(config_path)
    fig_str = plot_local_str_pruned_params(config)
    fig_unstr = plot_local_unstr_pruned_params(config)
    fig_global = plot_global_unstr_pruned_params(config)
    plt.show()