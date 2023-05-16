import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
import numpy as np

from copy import deepcopy


from NeuralNetwork import Model
from DataClassesJSON import ConfigData, PrunConfig, MethodPrunConfig




def _get_new_features(model: Model):
    features = []
    feature_indeces = []
    num_linear_modules = _count_linear_modules(model)
    
    first_layer= True
    counter_linear_module = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            counter_linear_module += 1
            
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy()
            if first_layer:
                first_layer = False
                not_zero = [True for _ in range(module.in_features)]
            else:
                w_col_nonzero = np.any((weight != 0), axis=0)
                not_zero = list(map(any, zip(w_row_nonzero, w_col_nonzero, b_row_nonzero)))

            b_row_nonzero = (bias != 0)
            w_row_nonzero = np.any((weight != 0), axis=1)

            features.append(_count_nonzero(not_zero))
            feature_indeces.append(not_zero)

            if counter_linear_module == num_linear_modules:
                features.append(module.out_features)
                feature_indeces.append([True for _ in range(module.out_features)])
    return features, feature_indeces


def _count_linear_modules(model: Model):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            count += 1
    return count


def _count_nonzero(is_zero: list[list[bool]]):
    count = 0
    for is_zero_val in is_zero:
        if is_zero_val:
            count += 1
    return count



def removeNodes(old_features: list, model: Model):
    new_features, feature_indeces = _get_new_features(model)
    if new_features == old_features:
        return old_features, model

    idx = 0
    new_model = Model(new_features)
    for old_module, new_module in zip(model.modules(), new_model.modules()):
        if isinstance(old_module, nn.Linear) and isinstance(new_module, nn.Linear):
            out_indices = np.array([feature_indeces[idx+1]], dtype=bool)
            in_indices = np.array([feature_indeces[idx]], dtype=bool)
            idx += 1

            indices = torch.from_numpy(in_indices & out_indices.T)
            with torch.no_grad():
                new_module.weight = nn.parameter.Parameter(
                    torch.reshape(
                        old_module.weight[indices], (new_module.out_features, new_module.in_features)
                        )
                    )
                new_module.bias = nn.Parameter(old_module.bias[out_indices])
    return new_features, new_model



def removePruningReparamitrizations(
    model: Model, 
    make_sparse: bool = False
):
    """Removes masks and prune the actual weights and bias. 
    Remove weight_orig and bias_orig and the hooks. 

    INPUTS
    ------
        ``model`` is the pruned model. 

        ``make_sparse`` is True if the weights and bias should be made sparse.  

    RETURN
    ------
        ``model`` is the model with the mask applyed to the weight. 
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
                if make_sparse:
                    module.weight.to_sparse()
            except:
                pass
            try:
                prune.remove(module, "bias")
                if make_sparse:
                    module.bias.to_sparse()
            except:
                pass
    else:
        return model


    
def calc_iter_amount(iter: int, iterations: int, amount, schedule: str):
    if schedule == 'exp':
        multipl = 1 / (1-np.exp(-1))
        return amount*multipl*(1-np.exp(- (iter + 1) / iterations))# ((amount*100)**((iter + 1) / iterations)) / 100
    if schedule == 'binom':
        return 1 - (1-amount)**((iter + 1) / iterations)
    if schedule == 'lin':
        return amount*((iter + 1) / iterations)



def getSparsity(
    model: Model, 
    bias_or_weight: str, 
    print_sparsity=True,
    print_only_global_sp=False
) -> tuple[pd.DataFrame, float]:
    """Calculates the sparsity of every module and the global sparsity 
    of the whole network. Optionaly print the sparsity. 

    INPUTS
    ------
        ``pruned_model`` is the model and needed for the modules. 

        ``bias_or_weight`` is the string with 'bias' or 'weight'. 

        ``print_sparsity`` is True if the sparsity should be printed. 

    RETURN
    ------
        ``sparsity_frame`` is a frame with module names and it's sparsitys. 

        ``global_sparsity`` is the sparsity of the whole network. 
    """
    sparse_elements_arr = []
    nelements_arr = []
    module_sparsity = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            param_dict = {
                'weight': module.weight,
                'bias': module.bias
            }
            if bias_or_weight == 'both':
                parameters = param_dict.values()
            else:
                parameters = [param_dict[bias_or_weight]]
                
            for parameter in parameters:
                sparse_elements = float(torch.sum(parameter == 0))
                nelements = float(parameter.numel())
                sparse_elements_arr.append(sparse_elements)
                nelements_arr.append(nelements)

                sparsity = 100 * sparse_elements / nelements if nelements != 0 else 0
                module_sparsity.append({
                    'module_name': name, 
                    'sparsity': sparsity
                })
    sparsity_frame = pd.DataFrame(module_sparsity)

    global_sparsity = 100. * float(np.sum(sparse_elements_arr)) / float(np.sum(nelements_arr))

    if print_sparsity and not print_only_global_sp:
        printSparsity(bias_or_weight, sparsity_frame)
        printGlobalSparsity(global_sparsity)
    if print_only_global_sp:
        printGlobalSparsity(global_sparsity)
    return sparsity_frame, global_sparsity

    

def printSparsity(
    bias_or_weight: str, 
    sparsity_frame: pd.DataFrame
):
    """Prints the sparsity of every module of the model. 

    INPUTS
    ------
        ``bias_or_weight`` is the string with 'bias' or 'weight'. 

        ``sparsity_frame`` is a frame with module names and it's sparsitys. 
    """
    for module_name, sparsity in zip(sparsity_frame['module_name'], sparsity_frame['sparsity']):
        print(f'Sparsity in {module_name}.{bias_or_weight:}: {sparsity:.2f}%')
    


def printGlobalSparsity(global_sparsity: float):
    """Prints the global sparsity of the model. 

    INPUTS
    ------
        ``global_sparsity`` is the sparsity of the whole network. 
    """
    print(f'Global sparsity: {global_sparsity:.2f}%')
    



def getPrunConfig(loaded_config: ConfigData, method_prun_config: PrunConfig | MethodPrunConfig) -> ConfigData:
    """Calculates the pruning configurations 

    INPUTS
    ------
        ``loaded_config`` is the save directory of the config

        ``method_prun_config`` 

    RETURN
    ------
        ``prun_config`` 
    """
    prun_config = deepcopy(loaded_config)
    prun_config.layer = method_prun_config.layer
    prun_config.nodes = method_prun_config.nodes 
    prun_config.networks = method_prun_config.networks
    return prun_config



def restorePrunMask(prun_model: Model, init_model: Model):
    for (prun_name, prun_module), (init_name, init_module) in zip(prun_model.named_modules(), init_model.named_modules()):
        if isinstance(prun_module, nn.Linear) and isinstance(init_module, nn.Linear):
            # weights
            if hasattr(prun_module, 'weight_mask'):
                weight_mask = prun_module.weight_mask.data.detach().clone()
                prune.custom_from_mask(init_module, name='weight', mask=weight_mask)

            # bias
            if hasattr(prun_module, 'bias_mask'):
                bias_mask = prun_module.bias_mask.data.detach().clone()
                prune.custom_from_mask(init_module, name='bias', mask=bias_mask)
    return init_model