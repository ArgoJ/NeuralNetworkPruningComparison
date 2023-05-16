import numpy as np
import torch


def calcR2Torch(loss: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Function to calculate the R² Score in a tensor. ->
    y_bar = 1/N * sum(y_i) ,  r_squared = 1 - sum((y_i - pred_i)^2) / sum((y_i - y_bar)^2) 

    INPUTS
    ------
        ``loss`` is an array of the loss.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``r_squared`` is the R² Score 
    """
    y_bar = torch.mean(true[:])
    sq_dominator = torch.square(true[:]  - y_bar)   
    summ_dominator = torch.sum(sq_dominator[:])
    r_squared = 1 - loss*len(true)/summ_dominator
    return r_squared


def calcR2(pred: np.ndarray, true: np.ndarray) -> float:
    """Function to calculate the R² Score. ->
    y_bar = 1/N * sum(y_i) ,  r_squared = 1 - sum((y_i - pred_i)^2) / sum((y_i - y_bar)^2) 

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``r_squared`` is the R² Score 
    """
    y_bar = np.mean(true[:])
    sq_counter = np.square(true[:] - pred[:])
    summ_counter = np.sum(sq_counter[:])
    sq_dominator = np.square(true[:]  - y_bar)   
    summ_dominator = np.sum(sq_dominator[:])
    r_squared = 1 - summ_counter/summ_dominator
    return r_squared



def calcMSE(pred: np.ndarray, true: np.ndarray) -> float:
    """Function to calculate mean squared error. ->
    mse = 1/N * sum(square(y_i - pred_i)) 

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``mse`` is the mean squared error
    """
    sq = np.square(true[:] - pred[:])
    mse = np.mean(sq[:])
    return mse



def calcRMSE(pred: np.ndarray, true: np.ndarray) -> float:
    """Function to calculate root mean squared error. ->
    rmse = root(1/N * sum(square(y_i - pred_i)))

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``rmse`` is the root mean squared error
    """
    sq = np.square(true[:] - pred[:])
    mse = np.mean(sq[:])
    rmse = np.sqrt(mse)
    return rmse



def calcMAE(pred: np.ndarray, true: np.ndarray) -> float:
    """Function to calculate mean absolute error. ->
    mae = 1/N * sum(|y_i - pred_i|) 

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``mae`` is the mean absolute error
    """
    ab = np.abs(true[:] - pred[:])
    mae = np.mean(ab[:])
    return mae



# function to calculate MAPE
# mape = 1/N * sum(|y_i - pred_i| / |y_i|)
def calcMAPE(pred: np.ndarray, true: np.ndarray) -> float:
    """Function to calculate mean absolute percentage error. ->
    mape = 1/N * sum(|y_i - pred_i| / |y_i|)

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``mape`` is the mean absolute percentage error.
    """
    pc = np.abs(true[:] - pred[:]) / np.abs(true[:])
    mape = np.mean(pc[:])
    return mape



# function to calc ME
# me = 1/N * sum(pred_i - y_i)
def calcME(pred: np.ndarray, true: np.ndarray) -> float:
    """Function to calculate mean error. ->
    me = 1/N * sum(pred_i - y_i)

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``me`` is the mean error.
    """
    dif = pred[:] - true[:]
    me = np.mean(dif)
    return me



# function to calculate accuracy
# accuracy = 1/N * True
# pred is True if it is in range of +-0.5% of y
def calcAccuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Function to calculate the accuracy  as if the points are 
    in range of the 5% band around the true function. ->
    accuracy = 1/N * True, pred is True if it is in range of +-0.5% of y

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``accuracy`` is the percentage of the number of 
        the true values to the number of all values.
    """
    threshold = true * 0.05
    lower_lim = true - threshold
    upper_lim = true + threshold
    limits = np.logical_and((lower_lim < pred[:]), (pred[:] < upper_lim))
    acc = np.sum(limits[:])
    accuracy = acc/len(pred[:])
    return accuracy



# functoin to calculate the absolute error of a matrix
# 
def calcAbsError(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Function to calculate the absolute error of every point. ->
    error = abs(original_matrix - predicted_matrix)

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``abs_error`` is the absolute error of every point as an array.
    """

    abs_error = np.abs(pred - true)
    return abs_error



def calcError(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Function to calculate the error of every point. ->
    error = original_matrix - predicted matrix 

    INPUTS
    ------
        ``pred`` is an array of the predicted values of the model.

        ``true`` is an array of the true values.

    RETURN
    ------
        ``error`` is the error of every point as an array.
    """
    error = pred - true
    return error