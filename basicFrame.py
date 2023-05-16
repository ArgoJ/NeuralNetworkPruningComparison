import pandas as pd
import numpy as np


from Constants import *


class HyperFrame(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class InTrainEvals(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class Metrics(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class MeanMetrics(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class IterMetrics(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class IterMeanMetrics(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class IterSparsity(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class Performances(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

class MeanPerformances(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
class Sparsities(pd.DataFrame):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()




def getDefaultHyperFrame(row_size: int, array) -> HyperFrame:
    """Creates a Dataframe with zeros of the shape (networks, col_len).

    INPUTS
    ------
        ``row_size`` is the size of the rows. (e.g. networks)

    RETURN
    ------
        ``hyper_frame`` is the hyperparameter frame with the shape (networks, col_len).
    """
    return getFrame(HyperFrame, row_size, HYPER_NAMES, array=array)



def getDuringTrainFrame(
    changed_hyperparam: pd.Series,
    runs: int,
    epochs: int, 
    np_array: np.ndarray = None
) -> InTrainEvals:
    """Creates a Dataframe with zeros or the given array and the columnnames
    'R²', 'RMSE', 'MAE'.

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``runs`` is the size of the rows. (e.g. runs)
        
        ``epochs`` 
        
        ``np_array`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        InTrainEvals, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series(DURING_TRAIN_ENV_NAMES), 
        row_level_2 = pd.Series([*range(1, runs+1)], name='run'), 
        col_level_2 = pd.Series(DURING_TRAIN_NAMES),
        row_level_3 = pd.Series([*range(1, epochs+1)], name='epochs'),
        np_array = np_array
    )



def getMetricFrame(
    changed_hyperparam: pd.Series,
    runs: int,
    np_array: np.ndarray = None
) -> Metrics:
    """Creates a Dataframe with zeros or the given array and the columnnames
    'R²', 'RMSE', 'MAE'.

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``runs`` is the size of the rows. (e.g. runs)

        ``np_array`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        Metrics, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series(ENVIROMENT_NAMES), 
        row_level_2 = pd.Series([*range(1, runs+1)], name='run'), 
        col_level_2 = pd.Series(METRIC_NAMES), 
        np_array = np_array
    )



def getMeanMetricsFrame(
    changed_hyperparam: pd.Series, 
    np_array: np.ndarray = None
) -> MeanMetrics:
    """Creates a Dataframe with zeros or the given array and the columnnames
    'R²', 'RMSE', 'MAE'.

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``np_metrics`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        MeanMetrics, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series(ENVIROMENT_NAMES), 
        row_level_2 = pd.Series(MEAN_NAMES, name='means'), 
        col_level_2 = pd.Series(METRIC_NAMES), 
        np_array=np_array
    )



def getIterMetricsFrame(
    changed_hyperparam: pd.Series,
    runs: int, 
    iterations: int, 
    np_array: np.ndarray = None
) -> IterMetrics:
    """Creates a Dataframe with zeros or the given array and the columnnames
    'R²', 'RMSE', 'MAE'.

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``row_size`` is the size of the rows. (e.g. networks)

        ``np_metrics`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        IterMetrics, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series(ENVIROMENT_NAMES), 
        row_level_2 = pd.Series([*range(1, runs+1)], name='run'), 
        col_level_2 = pd.Series(METRIC_NAMES), 
        row_level_3 = pd.Series([*range(iterations+1)], name='iterations'),
        np_array = np_array
    )


def getIterMeanMetricsFrame(
    changed_hyperparam: pd.Series, 
    iterations: int, 
    np_array: np.ndarray = None
) -> IterMeanMetrics:
    """Creates a Dataframe with zeros or the given array and the columnnames
    'R²', 'RMSE', 'MAE'.

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``row_size`` is the size of the rows. (e.g. networks)

        ``np_metrics`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        IterMeanMetrics, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series(ENVIROMENT_NAMES), 
        row_level_2 = pd.Series([*range(iterations)], name='iterations'), 
        col_level_2 = pd.Series(METRIC_NAMES), 
        row_level_3 = pd.Series(MEAN_NAMES, name='means'),
        np_array = np_array
    )
    


def getIterSparsityFrame(
    changed_hyperparam: pd.Series,
    runs: int, 
    iterations: int, 
    np_array: np.ndarray = None
) -> IterSparsity:
    """Creates a Dataframe with zeros or the given array and the columnnames
    'R²', 'RMSE', 'MAE'.

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``row_size`` is the size of the rows. (e.g. networks)

        ``np_metrics`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        IterSparsity, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series(SPARSITY_NAMES), 
        row_level_2 = pd.Series([*range(1, runs+1)], name='run'), 
        row_level_3 = pd.Series([*range(iterations+1)], name='iterations'),
        np_array = np_array
    )


def getSparsitiesFrame(
    changed_hyperparam: pd.Series,
    runs: int,
    np_array: np.ndarray = None
) -> Metrics:
    """Creates a Dataframe with zeros or the given array and the columnnames
    'R²', 'RMSE', 'MAE'.

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``runs`` is the size of the rows. (e.g. runs)

        ``np_array`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        Sparsities, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series(['sparsities', 'total_sparsities']),
        row_level_2 = pd.Series([*range(1, runs+1)], name='run'), 
        np_array = np_array,
    )


def getPerformanceFrame(
    changed_hyperparam: pd.Series,
    runs: int,
    np_array: np.ndarray = None
) -> Performances:  
    """Creates a Dataframe with....

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``runs`` is the size of the rows. (e.g. networks)

        ``np_metrics`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        Performances, 
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series((*PERFORMANCE_NAMES, *METRIC_NAMES)), 
        row_level_2 = pd.Series([*range(1, runs+1)], name='run'),
        np_array = np_array
    )


def getMeanPerformanceFrame(
    changed_hyperparam: pd.Series,
    np_array: np.ndarray = None
) -> MeanPerformances:  
    """Creates a Dataframe with....

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``runs`` is the size of the rows. (e.g. networks)

        ``np_metrics`` is a possible arrray that's inserted in the dataframe.

    RETURN
    ------
        ``metrics_frame`` is the metrics frame with the shape (networks, col_len).
    """
    return get3DFrame(
        MeanPerformances,
        row_level_1 = changed_hyperparam, 
        col_level_1 = pd.Series((*PERFORMANCE_NAMES, *METRIC_NAMES)), 
        row_level_2 = pd.Series(MEAN_NAMES, name='means'),
        np_array = np_array
    )



def getFrame(instance: pd.DataFrame, row_size: int, column_names: list[str], array: np.ndarray = None) -> pd.DataFrame:
    """"Creates a Dataframe with zeros or the given array and the columnnames 
    of the shape (networks, col_len).

    INPUTS
    ------
        ``row_size`` is the size of the rows. (e.g. networks)

        ``column_names`` is a list of column name strings in it. 

        ``np_array`` is a possible array thats filled in the dataframe.

    RETURN
    ------
        ``frame`` is the frame with the shape (networks, col_len).
    """
    fill_array = np.zeros((row_size, len(column_names))) if array is None else array
    frame = instance(
            fill_array,
            columns=column_names,
            dtype=object)
    return frame



def get3DFrame(
    instance: pd.DataFrame,
    row_level_1: pd.Series,
    col_level_1: pd.Series,
    row_level_2: pd.Series, 
    col_level_2: pd.Series = None, 
    row_level_3: pd.Series = None,
    np_array: np.ndarray = None
) -> pd.DataFrame:
    """"Creates a Dataframe with zeros or the given array and the columnnames 
    of the shape (networks*runs, col_len).

    INPUTS
    ------
        ``changed_hyperparam`` is a Series of the changed hyperparameter. 

        ``runs`` runs

        ``column_names`` is a list of column name strings in it. 

        ``np_array`` is a possible array thats filled in the dataframe.

    RETURN
    ------
        ``frame`` is the frame with the shape (networks, col_len).
    """
    if row_level_3 is not None:
        indexes = pd.MultiIndex.from_product(
            [row_level_1.to_list(), row_level_2.to_list(), row_level_3.to_list()],
            names = [row_level_1.name, row_level_2.name, row_level_3.name]
        )
    else:
        indexes = pd.MultiIndex.from_product(
            [row_level_1.to_list(), row_level_2.to_list()], 
            names = [row_level_1.name, row_level_2.name]
        )
    
    if col_level_2 is not None:
        columns = pd.MultiIndex.from_product(
            [col_level_1.to_list(), col_level_2.to_list()],
            names = [col_level_1.name, col_level_2.name]
        ) 
    else: 
        columns = col_level_1

    fill_array = np.zeros(
        (len(indexes), len(columns))
    ) if np_array is None else np_array

    frame = instance(
            fill_array,
            index=indexes,
            columns=columns,
            dtype=float)
    return frame