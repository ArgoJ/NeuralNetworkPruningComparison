import numpy as np
import pandas as pd
import os
import torch
import timeit
import sys
import copy
from pyx import *

from fvcore.nn import FlopCountAnalysis, flop_count_table

# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Illustrations.ModelGraph import makeGraph
from NeuralNetwork import Model, _profileModelPreds, get_features
import torch.nn as nn
from collections import OrderedDict
import torch.nn.utils.prune as prune

import basicFrame
from Pruning.PruningMethods.MagnitudePruning import l1_unstructured_value_model, ln_structured_fixed_amount, ln_structured_value_model, global_unstructured_fixed_amount, l1_unstructured_fixed_amount, structured_next_params
import Saves
import Loads
from Pruning.PrunHelpers import getSparsity, removePruningReparamitrizations, removeNodes
from Helpers import selectDevice
from Illustrations.ModelDraws import makeDraws

curr_path = os.path.dirname(__file__)

PRUN_TYPES = {
    'l1_unstructured_amount': l1_unstructured_fixed_amount,
    'l1_unstructured_value': l1_unstructured_value_model,
    'ln_structured_amount': ln_structured_fixed_amount,
    'ln_structured_value': ln_structured_value_model,
    'global_unstructured_amount': global_unstructured_fixed_amount,
}

use_gpu_input = input('To use GPU, type in \'y\'.\n')
device = "cuda" if torch.cuda.is_available() and use_gpu_input=='y' else "cpu"


def compare_functions():
    loop = 1000
    degree = 1
    inputs = 2
    nodes = 15
    layer = 2
    outputs = 1
    data  = torch.rand(10000, 2, device=device)*100

    torch.random.manual_seed(364)

    model = Model(get_features(degree, inputs, layer, nodes, outputs))
    model = model.to(device)

    model.eval()
    getSparsity(model, 'weight')

    unprun_c = makeDraws(model)
    unprun_c.writeSVGfile(os.path.join(curr_path, 'rect_unpruned'))

    # dot = makeGraph(model, show_source=False)
    # dot.format = 'svg'
    # dot.render(os.path.join(curr_path, 'Graph_unprune'), view=True)

    time_pred = timeit.timeit(lambda: model(data), globals=globals(), number=loop)
    flops_unpr = FlopCountAnalysis(model, data)

    unpruned_save_path = os.path.join(curr_path, 'model_unpruned___0.pt')
    torch.save(model, unpruned_save_path)
    unpr_size = os.path.getsize(unpruned_save_path)
     




    prun_model = copy.deepcopy(model)
    prun_model = PRUN_TYPES['nodes_ln_structured_amount'](
        model, 
        'weight', 
        0.16
    )
    # prun_model = prune_nodes_structured(
    #     ln_structured_fixed_amount, 
    #     prun_model,
    #     'weight', 
    #     0.16)
    rep_model = removePruningReparamitrizations(prun_model, make_sparse=False)


    getSparsity(rep_model, 'weight')

    zeros_c = makeDraws(rep_model)
    zeros_c.writeSVGfile(os.path.join(curr_path, 'rect_zeros'))

    # dot = makeGraph(rep_model, show_source=False)
    # dot.format = 'svg'
    # dot.render(os.path.join(curr_path, 'Graph_w_zeros'), view=True)

    time_pred_zeros = timeit.timeit(lambda: rep_model(data), globals=globals(), number=loop)
    flops_rep = FlopCountAnalysis(rep_model, data)

    zeros_save_path = os.path.join(curr_path, 'model_prun_zeros_1.pt')
    torch.save(rep_model, zeros_save_path)
    rep_size = os.path.getsize(zeros_save_path)






    rem_nodes_model = removeNodes(rep_model)
    getSparsity(rem_nodes_model, 'weight')

    rem_c = makeDraws(rem_nodes_model)
    rem_c.writeSVGfile(os.path.join(curr_path, 'rect_removed'))

    # dot = makeGraph(rem_nodes_model, show_source=False)
    # dot.format = 'svg'
    # dot.render(os.path.join(curr_path, 'Graph_r_zeros'), view=True)

    rem_nodes_model.eval()
    time_pred_removed = timeit.timeit(lambda: rem_nodes_model(data), globals=globals(), number=loop)
    flops_rem = FlopCountAnalysis(rem_nodes_model, data)

    rem_save_path = os.path.join(curr_path, 'model_prun_zeros_2.pt')
    torch.save(rem_nodes_model, rem_save_path)
    rem_zeros_size = os.path.getsize(rem_save_path)



    # print(f'Params unpruned: {list(model.named_parameters())}')
    # print(f'Params zerod: {list(rep_model.named_parameters())}')
    # print(f'Params removed: {list(rem_nodes_model.named_parameters())}')

    _profileModelPreds(model, data)
    _profileModelPreds(rep_model, data)
    _profileModelPreds(rem_nodes_model, data)

    print(f'Predictions params unpruned: {time_pred / loop}')
    print(f'Predictions params zeroed: {time_pred_zeros / loop}')
    print(f'Predictions nodes removed: {time_pred_removed / loop}')

    print(f'Memory params unpruned: {unpr_size} Bytes')
    print(f'Memory params zeroed: {rep_size} Bytes')
    print(f'Memory nodes removed: {rem_zeros_size} Bytes')

    print(f'Flops params unpruned: \n{flop_count_table(flops_unpr)}')
    print(f'Flops params zeroed: \n{flop_count_table(flops_rep)}')
    print(f'Flops nodes removed: \n{flop_count_table(flops_rem)}')


compare_functions()


def my_canvas():
    degree = 1
    inputs = 2
    nodes = 7
    layer = 3
    outputs = 1

    torch.random.manual_seed(15)

    model = Model(get_features(degree, inputs, layer, nodes, outputs))
    model = model.to(device)
    model.eval()
    getSparsity(model, 'weight')
     

    prun_model = copy.deepcopy(model)
    prun_model = ln_structured_fixed_amount(prun_model, 'weight', 0.16)
    prun_model = structured_next_params(prun_model)
    rep_model = removePruningReparamitrizations(prun_model, make_sparse=False)
    getSparsity(rep_model, 'weight')


    c = makeDraws(prun_model)
    c.writeSVGfile('rect')



# my_canvas()




print()