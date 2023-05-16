import numpy as np


import Helpers

from NeuralNetwork import Model, batchTrain, evalModel
from DataClassesJSON import ConfigData, MethodPrunConfig
from Constants import METRIC_NAMES

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Pruning.Const_Prun_Types import PRUN_TYPES
from Pruning.PrunHelpers import restorePrunMask, getSparsity, calc_iter_amount, removePruningReparamitrizations
from Pruning.PruningMethods.MagnitudePruning import structured_next_params




def prun_and_retrain(
    model: Model,
    method_config: MethodPrunConfig,
    model_config: ConfigData,
    data_cls: Data2d | Data3d,
    features: list[int], 
    run_index: int,
    iter_idx: int = 0,
    amount = None,
    init_after_prun: bool = False, 
    init_type: str = 'random', # or 'original'
    print_something = False
):
    # GET DATA =================================================
    train_data = data_cls.getTrainData()
    validation_data = data_cls.getValidationTensor()
    test_data = data_cls.getTestData()

    # set amount
    amount = method_config.amount if amount is None else amount

    # get global sparsity for later check if pruned
    _, global_sparsity_unpruned = getSparsity(model, method_config.bias_or_weight, print_sparsity=False)


    # PRUN MODEL ===============================================
    model = PRUN_TYPES[method_config.type](
        model, 
        method_config.bias_or_weight, 
        amount
    )
    
    if iter_idx+1 >= method_config.iterations and method_config.last_iter_npp:
        model = structured_next_params(model)

    _, global_sparsity_pruned = getSparsity(model, method_config.bias_or_weight, print_sparsity=False)

    # NOTHING IS PRUNED RETURN =================================
    # returns model without new param init or training
    if global_sparsity_unpruned == global_sparsity_pruned:
        evals = np.append(
            evalModel(model, train_data), 
            evalModel(model, test_data)
        )
        return model, evals, global_sparsity_pruned

    if init_after_prun:
        if init_type == 'random':
            init_model = Model(features, manu_seed=run_index)
        elif init_type == 'original':
            init_model = Model(features)

        assert 'init_model' in locals(), 'init_model is not inizialised!'
        
        if Helpers.use_cuda:
            model.cuda()
            init_model.cuda()
        model = restorePrunMask(model, init_model)


    ## TRAIN PROCESS ===========================================
    train_loader = data_cls.getDataLoader(
        train_data, 
        model_config.batch_size, 
        loader_reprod=False, 
        run=run_index
    )

    model, _ = batchTrain(
        model, 
        train_loader, 
        validation_data, 
        model_config.epochs, 
        model_config.learning_rate,
        data_cls.getTrueDataSize()[0],
        print_log=False
    )


    ## GET END SPARSITY ========================================
    _, global_sparsity = getSparsity(
        model, method_config.bias_or_weight, print_sparsity=False, print_only_global_sp=print_something
    )
    evals = np.append(
        evalModel(model, train_data), 
        evalModel(model, test_data)
    )
    return model, evals, global_sparsity




def iter_prun_and_retrain(
    model: Model,
    method_config: MethodPrunConfig,
    model_config: ConfigData,
    data_cls: Data2d | Data3d,
    features: list[int], 
    run_index: int,
    init_after_prun: bool = False,
    init_type: str = 'random', # or 'original'
    print_something = False
):
    eval_iter = np.zeros((method_config.iterations+1, len(METRIC_NAMES)*2)) 
    sparsity = np.zeros((method_config.iterations+1, 1))

    # unpruned model evals
    eval_iter[0] = np.append(
        evalModel(model, data_cls.getTrainData()), 
        evalModel(model, data_cls.getTestData())
    )
    
    for iter in range(method_config.iterations):
        if print_something:
            print(f'Pruning iteration {iter+1}/{method_config.iterations}')

        dynamic_amount = calc_iter_amount(
            iter, method_config.iterations, method_config.amount, method_config.schedule
        ) if 0<method_config.amount<1 else method_config.amount
        
        if print_something:
            print(f'amount: {dynamic_amount*100:.2f}%')

        (model, evals, global_sparsity) = prun_and_retrain(
            model,
            method_config,
            model_config,
            data_cls,
            features, 
            run_index,
            iter_idx=iter,
            amount=dynamic_amount,
            init_after_prun=init_after_prun,
            init_type=init_type
        )

        if iter+1 != method_config.iterations:
            model = removePruningReparamitrizations(model)

        # print(f'buffers: {list(pruned_model.named_buffers())}')
        sparsity[iter+1] = global_sparsity
        eval_iter[iter+1] = evals
    return model, eval_iter, sparsity