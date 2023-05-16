import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from collections import OrderedDict
from itertools import chain


import MetricFunctions
import Helpers



## MODEL
class Model(nn.Module):
    """Own instance of torch.nn.Module"""
    def __init__(
        self, 
        features: list[int], 
        man_seed = 31
    ) -> None:
        """Inistialises a Model with given inputs, outputs, degree and hidden layer and node sizes.

        INPUTS
        ------
            ``features`` is the list of the features as ints for every layer. 

            ``manu_seed`` is the manual seed of the Model parameters, default is 31. 
        """
        super(Model, self).__init__()

        torch.manual_seed(man_seed)

        hidden_layer = len(features) - 2
        modules = list(chain.from_iterable([
            *[[
                (f'hidden_{i+1}', nn.Linear(features[i], features[i+1])),
                (f'relu_{i+1}', nn.ReLU())
            ] for i in range(hidden_layer)],
            [(f'output', nn.Linear(features[-2], features[-1]))]
        ]))
        self.seq = nn.Sequential(OrderedDict(modules))


    def forward(
        self, 
        input: torch.Tensor
    ) -> torch.Tensor:
        """Function predicts the output for given input with the initialized model.

        INPUTS
        ------
            ``input`` is a processed tensor of values the model predicts

        RETURN
        ------
            ``out`` is a tensor of the predictions of the model for the input
        """
        return self.seq(input)



def get_features(
    degree: int, 
    inputs: int, 
    hidden_layer: int, 
    nodes: int, 
    outputs: int
):
    """Gets the features of every layer in a list with 
    the same feature size for the hidden layers. 
    
    INPUTS
    ------
        ``degree`` is the degree of the model.

        ``inputs`` is the number of inputs of the model. 

        ``layer`` is the hidden layer size of the model. 

        ``nodes`` is the size of the hidden layer nodes. 

        ``outputs`` is the number of outputs of the model. 

    RETURN
    ------
        ``features`` is the list of the features for every layer. 
    """
    return [degree*inputs, *[nodes for _ in range(hidden_layer)], outputs]




## TRAINING
#
# training with batches
def batchTrain(
    model: Model, 
    train_loader: DataLoader, 
    val_data: tuple[torch.Tensor, torch.Tensor], 
    epochs: int, 
    learning_rate: float,
    train_data_size: int,
    print_log = True
):
    """Trains the model with the trainloader and evaluate it during training.

    INPUTS
    ------
        ``model`` is the model that's gonning to be trained.

        ``train_loader`` is the Dataloader for the training process.

        ``val_data`` is for validation during the training process (per epochs).

        ``epochs`` is the value of how often the model is trained with the same data.

        ``learning_rate`` is the value of the learning rate.

        ``print_log`` is True if the loss and epoch should be printed 10 times during training. 

    RETURN
    ------
        ``model`` is the trained model. 

        ``loss_data`` is a tuple of training loss and validation loss for every epoch. 

        ``r_sq_data`` is a tuple of the training and the validation R² Score for every epoch.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print_loss = int(epochs/(10 if epochs >= 10 else epochs))

    loss_train = np.zeros(epochs)
    r_sq_train = np.zeros(epochs)
    loss_val = np.zeros(epochs)
    r_sq_val = np.zeros(epochs)

    true_train = torch.zeros((train_data_size, 1))
    pred_train = torch.zeros((train_data_size, 1))

    # training loop
    for epoch in range(epochs):
        idx = 0
        true_train.fill_(0.)
        pred_train.fill_(0.)

        model.train()
        for input, truth in train_loader:
            pred = model(input)
            loss = criterion(pred, truth)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=1.)

            end_idx = idx + len(truth)
            true_train[idx:end_idx] = truth
            pred_train[idx:end_idx] = pred
            idx = end_idx

        with torch.no_grad():
            # load train data in lists
            loss_train[epoch]= criterion(pred_train, true_train).item()
            r_sq_train[epoch] = MetricFunctions.calcR2(
                true_train.detach().cpu().numpy(), 
                pred_train.detach().cpu().numpy()
            )

        model.eval()
        with torch.no_grad():
            # validate model during training
            input_val, truth_val = val_data
            pred_val = model(input_val)
            
            # load validation data in lists
            loss_val[epoch] = criterion(pred_val, truth_val).item()
            r_sq_val[epoch] = MetricFunctions.calcR2(
                pred_val.detach().cpu().numpy(), 
                truth_val.detach().cpu().numpy()
            )

        if (epoch+1) % print_loss == 0 and print_log:
            print(
                f'epoch {epoch+1:>5}/{epochs}', ' '*5, 
                f'loss = {np.mean(loss_train[epoch]):>6.2f}', ' '*5,
                f'loss_val = {loss_val[epoch]:>6.2f}'
            )
        
        # empty gpu cache
        torch.cuda.empty_cache()
    return model, np.stack((loss_train, r_sq_train, loss_val, r_sq_val), axis=1)




## TEST
#
# evaluate data
def evalModel(
    model: nn.Module, 
    data: tuple[torch.Tensor, torch.Tensor], 
    profile_pred: bool = False
) -> np.ndarray:
    """Function evaluates the model with R² Score and RMSE. 

    INPUTS
    ------
        ``model`` is the model for evaluation. 

        ``data`` is the processed data the model predicts. 

        ``profile_pred`` is True if the the predictions should be profiled. 

    RETURN
    ------
        ``eval`` is a list of the R² Score and RMSE for the model predictions. 
    """
    input_tensor, true_tensor = data
    pred = getModelPreds(model, input_tensor, profile_pred)
    true = true_tensor.detach().cpu().numpy()
    r2 = MetricFunctions.calcR2(pred, true)
    rmse = MetricFunctions.calcRMSE(pred, true)

    eval = np.array([r2, rmse])
    return eval



# get model predictions from input
def getModelPreds(
    model: nn.Module, 
    input: torch.Tensor | np.ndarray, 
    profile_pred: bool = False
) -> np.ndarray:
    """Function gets the input and its given prediction of the model.

    INPUTS
    ------
        ``model`` is the model that predicts values.

        ``input`` is an array of the input for the model. 
        
        ``profile_pred`` is True if the the predictions should be profiled.  

    RETURN
    ------
        ``input`` is an array of the input for the model.
        
        ``pred`` is an array of predictions of the model.
    """
    if type(input) is np.ndarray:
        input = torch.from_numpy(input).float()
        if Helpers.use_cuda:
            input = input.cuda()   

    model.eval()
    with torch.no_grad():
        pred = model(input).detach().cpu().numpy() if not profile_pred else _profileModelPreds(model, input)
    return pred



def _profileModelPreds(
    model: Model, 
    input: torch.Tensor
):
    """Function gets the input and its given prediction of the model.

    INPUTS
    ------
        ``model`` is the model that predicts values.

        ``input`` is an array of the input for the model. 

    RETURN
    ------
        ``input`` is an array of the input for the model.
        
        ``pred`` is an array of predictions of the model.
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, 
        record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            pred = model(input).detach().cpu().numpy()
        
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    return pred