from NeuralNetwork import Model
from DataClassesJSON import ConfigData, MethodPrunConfig

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Pruning.PruningMethods.PrunAndRetrain import iter_prun_and_retrain, prun_and_retrain



def lotteryTicket(
    model: Model,
    method_config: MethodPrunConfig,
    model_config: ConfigData,
    data_cls: Data2d | Data3d,
    features: list[int], 
    run_index: int,
    print_something: bool
):  
    if method_config.iterations > 1:
        return iter_prun_and_retrain(
            model, 
            method_config,
            model_config,
            data_cls,
            features, 
            run_index,
            init_after_prun=True,
            init_type='original',
            print_something=print_something
        )
    else:
        return prun_and_retrain(
            model, 
            method_config,
            model_config,
            data_cls,
            features, 
            run_index,
            init_after_prun=True,
            init_type='original',
            print_something=print_something
        )