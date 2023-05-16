import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np


import Helpers

from NeuralNetwork import Model, batchTrain, evalModel
from DataClassesJSON import ConfigData, MethodPrunConfig

from Pruning.PrunHelpers import getSparsity, removePruningReparamitrizations, calc_iter_amount

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d





#====================================================================================================
# UNSTRUCTURED
#====================================================================================================
class SynFlowUnstructured(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        super().__init__()
        prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        """Computes a mask where the value is set to zero 
        if score of the synaptic flow, which is
        S(Omega) = dR/dOmega * Omega ,
        is in the smales ``amount`` percentage or total values.
        """
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        score = torch.abs(t.grad * t)
        if nparams_toprune != 0: 
            topk = torch.topk(score.view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        return super(SynFlowUnstructured, cls).apply(
            module, name, amount=amount
        )


def local_unstructured_syn_flow(module: Model, param_name: str, amount):
    """

    INPUTS
    ------
        ``module`` is the module that should be pruned.

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``amount`` 
    RETURN
    ------
        ``module`` is the pruned module. 
    """
    SynFlowUnstructured.apply(module, param_name, amount=amount)
    return module



#====================================================================================================
# GLOBAL
#====================================================================================================
def global_unstructured_syn_flow(model: Model, param_name: str, amount):
    """

    INPUTS
    ------
        ``module`` is the module that should be pruned.

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``amount`` 
    RETURN
    ------
        ``module`` is the pruned module. 
    """
    parameters_to_prune = [
        (module, param_name) 
        for module in filter(lambda m: type(m) == nn.Linear, model.modules())
    ]
    scores = []
    for module, _ in parameters_to_prune:
        t = getattr(module, param_name)
        scores.append(torch.abs(t.grad * t).view(-1))

    score_vec = torch.cat(scores)
    mask = torch.ones_like(score_vec)

    tensor_size = score_vec.nelement()
    nparams_toprune = prune._compute_nparams_toprune(amount, tensor_size)
    prune._validate_pruning_amount(nparams_toprune, tensor_size)
    if nparams_toprune != 0: 
        topk = torch.topk(score_vec, k=nparams_toprune, largest=False)
        mask[topk.indices] = 0

    # apply mask
    pointer = 0
    for module, name in parameters_to_prune:
        param = getattr(module, name)
        num_param = param.numel()
        param_mask = mask[pointer : pointer + num_param].view_as(param)
        prune.custom_from_mask(module, name, mask=param_mask)
        pointer += num_param
    
    return model





#====================================================================================================
# BASIC SYNAPTIC FLOW
#====================================================================================================
def synFlow_pruning(model: Model, param_name: str, amount, structure: str):
    @torch.no_grad()
    def abs_params(model: Model):
        signs = {}
        for name, param in model.named_parameters():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def remove_abs_params(model: Model, signs):
        for name, param in model.named_parameters():
            param.mul_(signs[name.strip('_orig')])

    model.eval()

    # set parameters to its absolutes 
    signs = abs_params(model)

    # make the gradients with backpropagation
    input_len = next((module.in_features for module in model.modules() if isinstance(module, nn.Linear)), None)
    input = torch.ones((1, input_len), device='cuda' if Helpers.use_cuda else 'cpu')
    output = model(input)
    torch.sum(output).backward()

    # prune the network
    if structure == 'unstructured':
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'output' in name:
                    amount *= 0.5
                local_unstructured_syn_flow(module, param_name=param_name, amount=amount)
    elif structure == 'global':
        global_unstructured_syn_flow(model, param_name=param_name, amount=amount)
    else:
        raise(ValueError('Not the correct structure input!'))


    remove_abs_params(model, signs)
    return model



def iter_synFlow(model: Model, param_name: str, amount, structure: str, iterations: int):
    if iterations == 0:
        iterations = 1
    for iter in range(iterations):
        dyn_amount = calc_iter_amount(iter, iterations, amount, 'binom')
        model = synFlow_pruning(model, param_name, dyn_amount, structure)
        if iter+1 != iterations:
            model = removePruningReparamitrizations(model)
    return model





def synFlow_and_train(
    model: Model,
    method_config: MethodPrunConfig,
    model_config: ConfigData,
    data_cls: Data2d | Data3d,
    features: list[int], 
    run_index: int,
    print_something = False
):
    train_data = data_cls.getTrainData()
    validation_data = data_cls.getValidationTensor()
    test_data = data_cls.getTestData()

    structure = 'global' if 'global' in method_config.type else 'unstructured'     
    model = Model(features)
    if Helpers.use_cuda:
        model.cuda()

    model = iter_synFlow(
        model, method_config.bias_or_weight, method_config.amount, structure, method_config.iterations
    )

    train_loader = data_cls.getDataLoader(
        train_data, 
        model_config.batch_size, 
        loader_reprod=False, 
        run=run_index
    )

    batchTrain(
        model, 
        train_loader, 
        validation_data, 
        model_config.epochs, 
        model_config.learning_rate,
        data_cls.getTrueDataSize()[0],
        print_log=False
    )

    _, global_sparsity = getSparsity(
        model, method_config.bias_or_weight, print_sparsity=False, print_only_global_sp=print_something
    )
    evals = np.append(
        evalModel(model, train_data), 
        evalModel(model, test_data)
    )
    return model, evals, global_sparsity
