from dataclasses import dataclass



#====================================================================================================
# BASE CONFIG
#====================================================================================================
@dataclass
class ConfigData():
    """Dataclass for configs of the multi Neural Nets
    """
    networks: int
    runs: int
    loader_reprod: bool
    make_plots: bool
    train_size: int
    test_size: int
    degree: int
    inputs: int
    outputs: int
    layer: int | list[int]
    nodes: int | list[int]
    learning_rate: float
    epochs: int
    batch_size: int
    layer_step: int
    node_step: int
    mean: float
    std: float


@dataclass
class BaseConfig():
    """Dataclass for configs of the dimension
    """
    networks: int
    runs: int
    loader_reprod: bool
    make_plots: bool
    train_size: int
    test_size: int
    inputs: int
    outputs: int
    learning_rate: float
    epochs: int
    batch_size: int
    mean: float
    std: float


@dataclass
class ArchitectureConfig():
    """Dataclass for the config of the architectures. 
    """
    degree: int
    layer: int | list[int]
    nodes: int | list[int]
    layer_step: int
    node_step: int


#====================================================================================================
# PRUNING
#====================================================================================================
@dataclass
class PrunConfig:
    """Dataclass with the pruning configurations for magnitude pruning.
    """
    type: str
    networks: int
    layer: int | list[int]
    nodes: int | list[int]
    amount: float 
    bias_or_weight: str
    remove_nodes: bool
    prun_prev_params: bool
    prun_next_params: bool
    
    
@dataclass
class MethodPrunConfig(PrunConfig):
    """Dataclass with the pruning configurations for pruning 
    with a specific method like finetune or lotter ticket. 
    Also with iterations added. 
    """
    iterations: int
    schedule: str
    last_iter_npp: bool
    method: str


#====================================================================================================
# ALL PRUNING
#====================================================================================================
@dataclass
class AllPrunConfig:
    """Dataclass with the pruning configurations for magnitude pruning, 
    but pruns every network of folder.
    """
    type: str
    amount: float 
    bias_or_weight: str
    remove_nodes: bool
    prun_prev_params: bool
    prun_next_params: bool


@dataclass
class AllMethodPrunConfig(AllPrunConfig):
    """Dataclass with the pruning configurations for pruning 
    with a specific method like finetune or lotter ticket. 
    Also with iterations added. Pruns every network of folder.
    """
    iterations: int
    schedule: str
    last_iter_npp: bool
    method: str