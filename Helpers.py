import numpy as np
import torch
import pandas as pd
import os
import time
import threading
import multiprocessing

from time import perf_counter
from torch.utils.data import DataLoader

from CustomExceptions import DimensionException, NotExistingChangingParameter
from DataClassesJSON import ConfigData
from basicFrame import getDefaultHyperFrame, HyperFrame, IterMeanMetrics, getIterMeanMetricsFrame




#====================================================================================================
# TIME ESTIMATION CLASSES
#====================================================================================================
class PrintThread(threading.Thread):
    def __init__(self, queue: multiprocessing.Queue, **kwargs):
        super(PrintThread, self).__init__(target=self._bar_countdown, args=(queue, ), **kwargs)
        self._stop = threading.Event()
        self._pause = threading.Event()
        
    def stop(self):
        self._stop.set()
 
    def stopped(self):
        return self._stop.isSet()
    
    def pause(self):
        self._pause.set()

    def paused(self):
        return self._pause.isSet()

    def resume(self):
        self._pause.clear()

    def _bar_countdown(self, queue: multiprocessing.Queue):
        while not self.stopped():
            if not queue.empty():
                counter, total_counts = queue.get()
                while counter and not self.stopped() and not self._pause.isSet():
                    if not queue.empty(): break
                    loading_bar = _get_bar_str(counter, total_counts)
                    print('\r{} {}h {}m {}s{}'.format(loading_bar, *secToTime(counter), ' '*5), end='')
                    time.sleep(1)
                    counter -= 1
            else:
                time.sleep(0.1)



class TimeEstimation():
    def __init__(self, model_config: ConfigData, time_rows = 1) -> None:
        base_size = 1 + model_config.networks*model_config.runs
        self.times = np.zeros(base_size * time_rows)
        self.times[0] = perf_counter()
        self.times_processed = np.zeros_like(self.times)

        base_x = np.tile(np.linspace(0, base_size-1, base_size), (time_rows, 1))
        self.x = np.concatenate(base_x)
        
        self.time_rows = time_rows
        self.networks = model_config.networks
        self.runs = model_config.runs
        

    def calcTimeEstimation(
        self, 
        run_idx: int, 
        network_idx: int, 
        row_idx = 0,
        time_advertised = False,
    ):
        """Estimates the rest computing time and prints it.   

        INPUTS
        ------
            ``times`` are the times that every model needs to train.  

            ``run_idx`` is the number of the run.  

            ``network_idx`` is the number of the network. 

            ``runs`` is the total number of runs. 

            ``networks`` is the total number of networks. 

        RETURN
        ------
            ``estimated_time`` is the rest time in seconds. 
        """
        num_cols = 1 + self.networks*self.runs
        total_idxs = num_cols*self.time_rows 
        idx = 1 + run_idx + network_idx*self.runs + num_cols*row_idx

        time_const_idx = num_cols*row_idx
        col_idx = idx % num_cols
        if row_idx and col_idx==1:
            self.times[idx-1] = self.times[idx-2]
        
        self.times[idx] = perf_counter()
        self.times_processed[idx] = self.times[idx] - self.times[time_const_idx]

        deg = 1 if network_idx==0 else 2
        coef = np.polyfit(self.x[:idx+1], self.times_processed[:idx+1], deg)
        fn = np.poly1d(coef)
        
        one_row_time_estim = fn(num_cols) - fn(0)
        estimated_resttime = one_row_time_estim*(self.time_rows - (row_idx+1)) + fn(num_cols) - fn(idx)
        estimated_total_time = one_row_time_estim*self.time_rows

        if (total_idxs-1) != idx:
            self.printEstimateTime(estimated_resttime, estimated_total_time, time_advertised=time_advertised)
        else:
            total_time = self.times[-1] - self.times[0]
            self.printTotalTime(total_time, time_advertised=time_advertised)


    def printEstimateTime(
        self, estimated_resttime: float, estimated_total_time, time_advertised = False
    ):
        if time_advertised:
            print('Estimated rest computing time: {}h {}m {}s'.format(*secToTime(estimated_resttime)))
        else:
            global _queue, _thread
            if 0 < estimated_resttime < estimated_total_time:
                if _thread.paused(): _thread.resume()
                _queue.put((int(estimated_resttime), int(estimated_total_time)))
        

    def printTotalTime(self, total_time, time_advertised = False):
        global _thread
        _thread.pause()
        if not time_advertised:
            print('\r|{}|{}'.format('-' * 100, ' ' * 20 ))
        print('Total time: {}h {}m {}s'.format(*secToTime(total_time)))



def set_multi_thread():
    global _queue, _thread
    _queue = multiprocessing.Queue()
    _thread = PrintThread(queue=_queue)
    _thread.start()


def del_multi_thread():
    global _queue, _thread
    _thread.stop()
    del _thread
    del _queue
    



#====================================================================================================
# PRINTS
#====================================================================================================
def printEvals(eval_train, eval_test):
    """Function prints the evaluation data of one model.

    INPUTS
    ------
        ``eval_train`` is the train evaluation data of the model.

        ``eval_test`` is the test evaluation data of the model.
    """
    print(f'\nR² - Test: {eval_test[0]:.3f} - Train: {eval_train[0]:.3f}')
    print(f'RMSE - Test: {eval_test[1]:.2f} - Train: {eval_train[1]:.2f}')



def printMetrics(frame: pd.DataFrame, name: str):
    """Prints the metrics and hyperparameters of every run. 

    INPUTS
    ------
        ``frame`` is the metrics frame.  
        
        ``name``  is the name thats on top of the frame. 
    """
    print('-'*100)
    print(name)
    print(frame)



#====================================================================================================
# FUNKTIONS
#====================================================================================================
use_cuda = False
def selectDevice() -> torch.device:
    """Function lets you choose the device you want to move the models and data to. 
    Sets the torch.cuda.is_available to False. 

    RETURN
    ------
        ``device`` is the choosen device. 
    """
    global use_cuda
    if torch.cuda.is_available():
        use_gpu = input('To use GPU, type in \'y\'.\n')
        if 'y' in use_gpu:
            use_cuda = True

    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using {device} as device')
    return device


def inputParams():
    """Print input question for dimension and folder and 
    gets the inputs. 

    RETURN
    ------
        ``dimension`` is the dimension of the saved models. 

        ``time_folder`` is the folder of the saved models of type '%m_%d_%H_%M'. 
    """
    print('Type in the model you want to have.')
    dimension = int(input('What dimension is the model (2 or 3)?'))
    if not (dimension==2 or dimension==3):
        raise(DimensionException(dimension)) 
    time_folder = input('In which time folder is the model (%m_%d_%H_%M)?')
    return dimension, time_folder

        
def _get_bar_str(counter: int, total_counts: float):
    bars = int(100 * (1 - counter/total_counts))
    return '|' + '-' * bars + '|' + ' ' * (100 - bars) 


def secToTime(secs: float) -> tuple[int, int, int]:
    """Calculates the hours, minutes and seconds of only seconds.   

    INPUTS
    ------
        ``secs`` are the total seconds. 

    RETURN
    ------
        ``hours``, ``min`` and ``sec`` is the time in 3 ints. 
    """
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), int(s)


def largest_multiplier(number):
    """Finds the largest multiplier of a given number, but not itself. 

    INPUTS
    ------
        ``number`` is the number of what the fn should find the largest multiplier. 

    RETURN
    ------
        ``largest_multiplier`` is the largest multiplier of the given number. 
    """
    largest_multiplier = 0
    for i in range(number-1, 0, -1):
        if np.floor_divide(number, i) * i == number:
            largest_multiplier = i
            break
    return largest_multiplier 


def get_new_total_nodes(features: list[int]):
    return sum(features[1:-1])


def get_total_nodes(hyper_frame: pd.DataFrame):
    total_nodes = hyper_frame['layer'] * hyper_frame['nodes'] 
    total_nodes.name = 'nodes'
    return total_nodes



#====================================================================================================
# DIRECTORY STUFF
#====================================================================================================
def createDir(directory: str) -> bool:
    """Make a directory if it doesn't exist. 

    INPUTS
    ------
        ``directory`` is the path that is made.

    RETURN
    ------
        ``dir_exists`` is the true if the directory is created successfully.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'created dir: {directory}')
    return os.path.exists(directory)


def createTikzDir(base_dir: str):
    directory = os.path.join(base_dir, 'Tikz')
    createDir(directory)
    return directory


def rename_existing_dir(directory: str):
    while True:
        if os.path.exists(directory):
            directory += 'i'
        else:
            return directory



#====================================================================================================
# FRAME STUFF
#====================================================================================================
def get_hyper_index(hyper_frame: pd.DataFrame) -> int:
    """Gets the index in the hyperparameter frame where the hyperparameter 
    changes over the multi networks. 

    INPUTS
    ------
        ``hyper_frame`` is the frame of multi networks of type DataFrame.

    RETURN
    ------
        ``index`` is the index where the hyperparameter changes.
    """
    if len(hyper_frame.index)==1:
        return 0
    for index in range(len(hyper_frame.columns)):
        if (hyper_frame.iloc[0, index] != hyper_frame.iloc[:, index]).any():
            return index
    return None



def add_to_existing_frame(old_frame: pd.DataFrame, insert_frame: pd.DataFrame):
    """Adds the insert_frame to the old_frame and removes the duplicates in 
    the old frame. Also sort the indexes.

    INPUTS
    ------
        ``old_frame`` is the old frame where the new frame should be inserted. 

        ``insert_frame`` is the frame that should be inserted. 

    RETURN
    ------
        ``merged_frame`` is the frame with the new frame inserted. 
    """
    insert_frame = pd.concat([insert_frame, old_frame])
    return insert_frame[~insert_frame.index.duplicated()].sort_index()



def add_to_existing_frame_file(frame: pd.DataFrame, file_path:str, print_frame = False):
    """Adds a frame to an already saved frame. Saves the old frame as 
    'old_{file_name}.pkl'

    INPUTS
    ------
        ``frame`` is the frame that should be added to the already saved frame. 

        ``file_path`` is the path incl. the file_name, where the frame is saved. 

    RETURN
    ------
        ``frame_exists`` is True if the new frame is saved. 
    """
    if os.path.exists(file_path):
        loaded_frame: pd.DataFrame = pd.read_pickle(file_path)
        new_frame = add_to_existing_frame(loaded_frame, frame)
        if print_frame:
            print(f'frame: \n{new_frame}')

        directory, file_name = os.path.split(file_path)
        old_file_path = os.path.join(directory, f'old_{file_name}')
        old_file_path = find_new_pkl_path(old_file_path)
                    
        loaded_frame.to_pickle(old_file_path)
        if os.path.exists(file_path) and os.path.exists(old_file_path):
            os.remove(file_path)
    else:
        new_frame = frame
    new_frame.to_pickle(file_path)
    return os.path.exists(file_path)


def find_new_pkl_path(file_path: str, file_name_add = 1):
    if os.path.exists(file_path):
        new_file_path = file_path.replace('.pkl', '') + f'_({file_name_add})' + '.pkl'
        file_path = find_new_pkl_path(new_file_path, file_name_add=file_name_add+1)
    return file_path



def calcHyperparams(config: ConfigData) -> HyperFrame:
    """Calculates the frame with hyperparameters for every network. 

    INPUTS
    ------
        ``config`` is the configuration file of the models of type ConfigData

    RETURN
    ------
        ``hyper_frame`` is the frame of multi networks of type DataFrame
    """
    hyperparams = get_hyperlist(config)
    config.networks = len(hyperparams)
    return getDefaultHyperFrame(config.networks, hyperparams)



def get_hyperlist(config: ConfigData):
    """Is a list of the hyperparameters in the style of 
    [
        [layer_net0, nodes_net0], 
        ..., 
        [layer_netN, nodes_netN]
    ]

    INPUTS
    ------
        ``config`` is the configuration file of the models of type ConfigData 

    RETURN
    ------
        ``hyperparams`` is a list with the nodes and layers every network.
    """
    if type(config.nodes) is list:
        hyperparams = [
            [config.layer, nodes] for nodes in config.nodes
        ]
    elif type(config.layer) is list:
        hyperparams = [
            [layer, config.nodes] for layer in config.layer
        ]
    else:
        hyperparams = [
            [
                config.layer + config.layer_step*network,
                config.nodes + config.node_step*network
            ] for network in range(config.networks)
        ]
    return hyperparams



def add_new_index_chp(
    frame: pd.DataFrame, 
    new_chps: np.ndarray,
    print_frame = False
):
    """Adds a new index column with new changed hyper parameters (chp), infront 
    of the old chp. Sets values to integer if there are only floats with .0. 
    Renames the old chp to 'old_{name}'. 

    INPUTS
    ------
        ``frame`` is a frame with two indexes, where the 
        ``new_chp`` are set to the first position of the indexes.
    """
    old_chps = frame.index.get_level_values(0)
    chp_name = old_chps.name
    old_chps.name = f'old_{chp_name}'
    new_chps = new_chps.flatten()

    if all(new_chp.is_integer() for new_chp in new_chps):
        new_chps = new_chps.astype(int)
    new_chps = pd.Index(new_chps, name=chp_name)
    frame.index = pd.MultiIndex.from_arrays([new_chps, old_chps, frame.index.get_level_values(1)])  
    if print_frame:
        print(frame)



def get_new_changed_feature(
    features: list, 
    config: ConfigData
) -> int | float:
    """Gets the changing features, either the nodes of the hidden layers
    or the hidden layer. If the nodes are not the same over the hidden layers, 
    it will give a float. Otherwise it will give an int. 

    INPUTS
    ------
        ``features`` are the features of every layer in a list. 

        ``config`` is the config of the architecture of the models. 

    RETURN
    ------
        ``num_features`` is the number of the hidden layers features or hidden layers. 
    """
    if config.node_step!=0:
        if all(feature == features[1] for feature in features[1:-1]):
            return features[1]
        else:
            hidden_features = features[1:-1]
            return sum(hidden_features) / len(hidden_features)
    elif config.layer_step!=0:
        return len(features)-2
    else:
        raise(NotExistingChangingParameter(config.layer_step, config.node_step))



def add_Index_to_MultiIndex(multi_index: pd.MultiIndex, new_index: list | tuple, new_index_name: str, position: int = 0):
    """Adds to the MultiIndex one index row at the ``position`` with the name ``new_index_name``

    INPUTS
    ------
        ``multi_index`` is the already existent MultiIndex instance. 

        ``new_index`` are the new indexes for insertion. 

        ``new_index_name`` is the name of the new index. 

        ``position`` is the position where the new index should be inserted. 

    RETURN
    ------
        ``new_multi_index`` is the MultiIndex with the new indexes isnerted. 
    """
    array_len = len(multi_index.get_level_values(0))
    idx_mul = array_len/len(new_index)
    assert type(new_index) is list or type(new_index) is tuple, f'``new_index`` is not a list or tuple!'
    assert idx_mul.is_integer(), f'``new_index`` have to be divisible by {array_len}!'

    idx_mul = int(idx_mul)
    my_list  = [
        val for val in new_index for _ in range(idx_mul)
    ]
    new_index_list = pd.Index(
        my_list,
        name=new_index_name
    )
    multi_index_list = [multi_index.get_level_values(idx) for idx in range(len(multi_index.levels))]
    multi_index_list.insert(position, new_index_list)
    return pd.MultiIndex.from_arrays(multi_index_list)



#====================================================================================================
# GET MEAN FRAMES
#====================================================================================================
def getMeanFromRuns(input_frame: pd.DataFrame, frame_function):
    """Calculates the mean of the input in relation to the runs for every network. 

    !!Attention!!
    -------------
        DOES NOT WORK FOR ITER_MEAN...

    INPUTS
    ------
        ``input_frame`` is the dataframe from which the means are calculated. 

        ``frame_function`` is a function for the frame generation from basicFrames. 
        (e.g. getMeanPerformanceFrame or getMeanMetricsFrame)

    RETURN
    ------
        ``mean_frame`` is a frame of the mean of all random training runs.
    """
    changed_hyperparam = input_frame.index.get_level_values(0).unique()
    networks = len(changed_hyperparam)
    runs = len(input_frame.index.get_level_values(1).unique())
    np_input = input_frame.to_numpy(dtype=float).reshape((networks, runs, input_frame.shape[1]))

    mean_input = np_input.mean(axis=1)
    min_input = np_input.min(axis=1)
    max_input = np_input.max(axis=1)

    min_mean_dif = mean_input - min_input
    max_mean_dif = max_input - mean_input

    mean_input_errors = np.hstack((mean_input, min_mean_dif, max_mean_dif)).reshape((-1, input_frame.shape[1]))
    return frame_function(changed_hyperparam, np_array=mean_input_errors)



def getIterMeanMetrics(metrics: pd.DataFrame) -> IterMeanMetrics:
    """Calculates the mean of metric of every network over all runs. 

    INPUTS
    ------
        ``metrics`` 

    RETURN
    ------
        ``iter_mean_metrics`` 
    """
    changed_hyperparam = metrics.index.get_level_values(0).unique()
    networks = len(changed_hyperparam)
    runs = len(metrics.index.get_level_values(1).unique())
    iterations = len(metrics.index.get_level_values(2).unique())
    np_metrics = metrics.to_numpy(dtype=float).reshape((networks, runs, iterations, metrics.shape[1]))

    mean_metrics = np_metrics.mean(axis=1)
    min_metrics = np_metrics.min(axis=1)
    max_metrics = np_metrics.max(axis=1)

    min_mean_dif = mean_metrics - min_metrics
    max_mean_dif = max_metrics - mean_metrics

    mean_metrics_errors = np.dstack((mean_metrics, min_mean_dif, max_mean_dif)).reshape((-1, metrics.shape[1]))
    return getIterMeanMetricsFrame(changed_hyperparam, iterations, np_array=mean_metrics_errors)



#====================================================================================================
# MODEL STUFF
#====================================================================================================
def find_model(directory: str):
    """finds the file names in the ``directory`` that starts with either 
    'model' or 'nrn_model' ('nrn' is for  'not removed nodes')

    INPUTS
    ------
        ``directory`` is the folder as a string or path where the models are. 

    RETURN
    ------
        ``model_file_path`` is the file path of the models that starts with 'model'.  

        ``nrn_model_file_path`` is the file path of the models that starts with 'nrn_model'. 
    """
    file_names = os.listdir(directory)
    model_file_path = None
    nrn_model_file_path = None
    for file_name in file_names:
        if file_name.startswith('models'):
            model_file_path = os.path.join(directory, file_name)
        if file_name.startswith('nrn_models'):
            nrn_model_file_path = os.path.join(directory, file_name)
    
    assert model_file_path is not None or nrn_model_file_path is not None, f'{directory} has no models stored.'
    return model_file_path, nrn_model_file_path



#====================================================================================================
# COMPARISSON STUFF
#====================================================================================================
def comparissonIndexes(
    loaded_config: ConfigData, 
    prun_config: ConfigData
):
    """Calculates the indexes of the loaded hyperparameter frame, 
    to compare it with pruned hyperparameter. 

    INPUTS
    ------
        ``loaded_config`` is the loaded configuration file. 

        ``prun_config`` is the configuration file the pruned models. 

    RETURN
    ------
        ``indexes``
    """
    loaded_hyperlist = get_hyperlist(loaded_config)
    prun_hyperlist = get_hyperlist(prun_config)
    indexes = [idx for idx, loaded_element in enumerate(loaded_hyperlist) for prun_element in prun_hyperlist if loaded_element == prun_element]
    if len(indexes) != prun_config.networks:
        raise(ValueError())
    return indexes



def getSlicedMetrics(
    metrics: pd.DataFrame, 
    indexes: list
):
    """Slices the ``metrics`` to the ``indexes`` of the index level 0.  

    INPUTS
    ------
        ``metrics`` is the dataframe that is sliced. It has to have an index level.

        ``indexes`` are the indexes that should be maintained.

    RETURN
    ------
        ``sliced_metrics`` are the metrics of the indexes of index level 0. 
    """
    network_idx = metrics.index.get_level_values(0).unique()
    return metrics.loc[network_idx[indexes]] 
