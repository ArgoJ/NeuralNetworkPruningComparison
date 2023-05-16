import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from NeuralNetwork import Model




#====================================================================================================
# UNSTRUCTURED AMOUNT PRUNING
#====================================================================================================
def l1_unstructured_fixed_amount(model: Model, param_name: str, amount: float):
    """Prunes ``amount`` of the lowest L1 norms in one weight matrix 
    or bias vector. It is used for the whole ``model``, 
    but every module at once, with the same 'amount'.  
    This pruning technique is unstructured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned.

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``amount`` is the amount in percent or fixed connections in one module. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'output' in name:
                amount *= 0.5
            prune.l1_unstructured(module, name=param_name, amount=amount)
    return model




#====================================================================================================
# STRUCTURED AMOUNT PRUNING
#====================================================================================================
def ln_structured_fixed_amount(model: Model, param_name: str, amount: float, n: float, dim: int):
    """Prunes ``amount`` of the lowest L-inf norms of a vector in one 
    weight matrix or bias vector. It is used for the whole ``model``, 
    but every module at once, with the same 'amount'.  
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``amount`` is the amount in percent or fixed connections in one module. 

        ``n`` is the norm. 

        ``dim`` is the dimension of the structured pruning.  

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'output' in name: 
                amount = 0.0
            prune.ln_structured(module, name=param_name, amount=amount, n=n, dim=dim)
    return model


def linf_structured_fixed_amount(model: Model, param_name: str, amount: float):
    """Prunes ``amount`` of the lowest L-inf norms of a vector in one 
    weight matrix or bias vector. It is used for the whole ``model``, 
    but every module at once, with the same 'amount'.  
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``amount`` is the amount in percent or fixed connections in one module. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    return ln_structured_fixed_amount(model, param_name, amount, n=float('-inf'), dim=0)


def l2_structured_fixed_amount(model: Model, param_name: str, amount: float):
    """Prunes ``amount`` of the lowest L2 norms of a vector in one 
    weight matrix or bias vector. It is used for the whole ``model``, 
    but every module at once, with the same 'amount'.  
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``amount`` is the amount in percent or fixed connections in one module. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    return ln_structured_fixed_amount(model, param_name, amount, n=2.0, dim=0)


def l1_structured_fixed_amount(model: Model, param_name: str, amount: float):
    """Prunes ``amount`` of the lowest L1 norms of a vector in one 
    weight matrix or bias vector. It is used for the whole ``model``, 
    but every module at once, with the same 'amount'.  
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``amount`` is the amount in percent or fixed connections in one module. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    return ln_structured_fixed_amount(model, param_name, amount, n=1.0, dim=0)




#====================================================================================================
# GLOBAL PRUNING
#====================================================================================================
def global_unstructured_fixed_amount(model: Model, param_name: str, amount: float):
    """Prunes ``amount`` of the lowest L1 norms in the whole ``model`` of 
    bias or weights. This pruning technique is global and unstructured.  

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned.

        ``amount`` is the amount in percent or fixed connections in the model. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    parameters_to_prune = [
        (module, param_name) 
        for module in filter(lambda m: type(m) == nn.Linear, model.modules())
    ]
    prune.global_unstructured(
        parameters_to_prune, 
        pruning_method=prune.L1Unstructured, 
        amount=amount
    )
    return model




#====================================================================================================
# UNSTRUCTURED VALUE PRUNING
#====================================================================================================
class L1UnstructuredValued(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, norm_val):
        super().__init__()
        self.norm_val = norm_val

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        """Computes a mask where the value is set to zero 
        if the calculated L1 norm is lower than 'self.norm_val'. 
        """
        mask = default_mask.clone()
        indices = (torch.abs(t) <= self.norm_val).nonzero(as_tuple=True)
        mask[indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, norm_val):
        return super(L1UnstructuredValued, cls).apply(
            module, name, norm_val=norm_val
        )


def l1_unstructured_value(module: Model, param_name: str, norm_val: float):
    """Prunes the L1 norms that are lower than ``norm_val`` in the weights 
    or biases of one ``module``. This pruning technique is unstructured. 

    INPUTS
    ------
        ``module`` is the module that is pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum L1 norm value that is not pruned.   

    RETURN
    ------
        ``module`` is the pruned module. 
    """
    L1UnstructuredValued.apply(module, param_name, norm_val=norm_val)
    return module


def l1_unstructured_value_model(model: Model, param_name: str, norm_val: float):
    """Prunes the L1 norms that are lower than ``norm_val`` in the weights 
    or biases of one module for the whole ``model``. 
    This pruning technique is unstructured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum L1 norm value that is not pruned.   

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            l1_unstructured_value(module, param_name, norm_val)
    return model




#====================================================================================================
# STRUCTURED VALUE PRUNING
#====================================================================================================
class LnStructuredValued(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'

    def __init__(self, norm_val, n, dim=-1):
        super().__init__()
        self.norm_val = norm_val
        self.n = n
        self.dim = dim

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        """Computes a mask where the value is set to zero 
        if the calculated Ln norm of the chosen dim is lower than 'self.norm_val'. 
        """
        mask = default_mask.clone()
        
        dims = list(range(t.dim()))
        if self.dim < 0:
            self.dim = dims[self.dim]
        dims.remove(self.dim)
        
        norm: torch.Tensor = torch.norm(t, p=self.n, dim=dims)
        indices = (norm <= self.norm_val).nonzero()

        slc = [slice(None)] * len(t.shape)
        slc[self.dim] = indices
        mask[slc] = 0
        return mask

    @classmethod
    def apply(cls, module, name, norm_val, n, dim):
        return super(LnStructuredValued, cls).apply(
            module, name, norm_val=norm_val, n=n, dim=dim
        )


def ln_structured_value(module: Model, param_name: str, norm_val: float, n: float, dim: int):
    """Prunes the Ln norms that are lower than ``norm_val`` in the weights 
    or biases of one ``module``. This pruning technique is structured. 

    INPUTS
    ------
        ``module`` is the module that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned.   

        ``n`` is the norm as 1, 2 inf ...

        ``dim`` is the dimension that should be pruned (row -> 0, col -> 1)

    RETURN
    ------
        ``module`` is the pruned module. 
    """
    LnStructuredValued.apply(module, param_name, norm_val=norm_val, n=n, dim=dim)
    return module


def ln_structured_value_model(model: Model, param_name: str, norm_val: float, n: float, dim: int):
    """Prunes the Ln norms that are lower than ``norm_val`` in the weights 
    or biases of one module for the whole ``model``. 
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned. 

        ``n`` is the norm. 

        ``dim`` is the dimension of the structured pruning. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            ln_structured_value(module, param_name, norm_val, n=n, dim=dim)
    return model


def linf_structured_value(model: Model, param_name: str, norm_val: float):
    """Prunes the Ln norms that are lower than ``norm_val`` in the weights 
    or biases of one module for the whole ``model``. 
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned.   

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    return ln_structured_value_model(model, param_name, norm_val, n=float('-inf'), dim=0)


def l2_structured_value(model: Model, param_name: str, norm_val: float):
    """Prunes the Ln norms that are lower than ``norm_val`` in the weights 
    or biases of one module for the whole ``model``. 
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned.   

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    return ln_structured_value_model(model, param_name, norm_val, n=2.0, dim=0)


def l1_structured_value(model: Model, param_name: str, norm_val: float):
    """Prunes the Ln norms that are lower than ``norm_val`` in the weights 
    or biases of one module for the whole ``model``. 
    This pruning technique is structured. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned.   

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    return ln_structured_value_model(model, param_name, norm_val, n=1.0, dim=0)




#====================================================================================================
# NEXT OR PERVIOUS PARAMS
#====================================================================================================
class StructuredNextOrPrevMask(prune.BasePruningMethod):
    PRUNING_TYPE = 'global'

    def __init__(self, prev_mask: torch.Tensor, dim = -1, mask_dim = -1):
        """
            ``dim`` is for the dimension of the new mask

            ``mask_dim`` is 0 for previous and 1 for next
        """
        super().__init__()
        self.prev_mask = prev_mask
        self.dim = dim
        self.mask_dim = mask_dim

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        """Computes a mask where the weights and biases are pruned 
        because previous weights where pruned. (Removes complete nodes 
        with it's weights and biases) 
        """
        mask = default_mask.clone()

        zero_rows = torch.where(self.prev_mask.count_nonzero(dim=self.mask_dim) == 0)
        if zero_rows[0].shape[0] != 0:
            slc = [slice(None)] * len(t.shape)
            slc[self.dim] = torch.split(*zero_rows, 1)
            mask[slc] = 0
        return mask

    @classmethod
    def apply(cls, module, name, prev_mask, dim, mask_dim):
        return super(StructuredNextOrPrevMask, cls).apply(
            module, name, prev_mask, dim, mask_dim
        )


def structured_next_prev_params(module: Model, param_name: str, mask: torch.Tensor, dim: int, mask_dim: int):
    """Prunes complete nodes of one ``module``, if the ``prev_mask`` cut the the node of. 
    This pruning technique is structured and just an addition. 

    INPUTS
    ------
        ``module`` is the module that should be pruned.

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``mask`` is the weight mask of the previous or next module.  

        ``dim`` is the dimension of the zeros of the new mask generated 

        ``mask_dim`` is -1 if previous and 1 if next mask is given. 
    RETURN
    ------
        ``module`` is the pruned module. 
    """
    StructuredNextOrPrevMask.apply(module, param_name, prev_mask=mask, dim=dim, mask_dim=mask_dim)
    return module


#----------------------------------------------------------------------------------------------------
# NEXT PARAMS
def structured_next_params(model: Model):
    """Prunes complete nodes of the whole ``model``, if the ``prev_mask``  of one module 
    cut the the node of. This pruning technique is structured and just an addition. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``bias_or_weight`` chooses if the weights or the bias should be pruned. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    prev_mask = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if prev_mask is not None:
                structured_next_prev_params(module, 'weight', prev_mask, 1, 1)

            prev_mask = list(module.buffers())[0]

            if "output" not in name:
                structured_next_prev_params(module, 'bias', prev_mask, 0, 1)
    return model
    

def nodes_linf_structured(model: Model, param_name, amount):
    """Prunes complete nodes of the whole ``model`` with L-inf norm structured 
    pruning. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned.   

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    linf_structured_fixed_amount(model, param_name, amount)
    structured_next_params(model)
    return model


def nodes_l2_structured(model: Model, param_name, amount):
    """Prunes complete nodes of the whole ``model`` with L2 norm structured 
    pruning. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned.   

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    l2_structured_fixed_amount(model, param_name, amount)
    structured_next_params(model)
    return model


def nodes_l1_structured(model: Model, param_name, amount):
    """Prunes complete nodes of the whole ``model`` with l1 norm structured 
    pruning. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``param_name`` chooses if the weights or the bias should be pruned. 

        ``norm_val`` is the minimum Ln norm value that is not pruned.   

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    l1_structured_fixed_amount(model, param_name, amount)
    structured_next_params(model)
    return model




#----------------------------------------------------------------------------------------------------
# PERVIOUS PARAMS
def structured_prev_params(model: Model):
    """Prunes complete nodes of the whole ``model``, if the ``next_mask`` of one module 
    cut the node of. This pruning technique is structured and just an addition. 

    INPUTS
    ------
        ``model`` is the model that should be pruned. 

        ``bias_or_weight`` chooses if the weights or the bias should be pruned. 

    RETURN
    ------
        ``model`` is the pruned model. 
    """
    next_weight_masks = [
        list(module.buffers())[0] for module in model.modules() if isinstance(module, nn.Linear)
    ]
    next_mask_idx = 0
    iters = len(next_weight_masks) - 1
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            next_mask_idx += 1
            structured_next_prev_params(module, 'weight', next_weight_masks[next_mask_idx], 0, 0)
            structured_next_prev_params(module, 'bias', next_weight_masks[next_mask_idx], 0, 0)
            if iters == next_mask_idx:
                break
            
    return model