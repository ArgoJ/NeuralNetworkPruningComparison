from abc import ABC, abstractmethod
import torch
import numpy as np
import multiprocessing as mp
from time import perf_counter

from torch.utils.data import DataLoader, TensorDataset


import Helpers
from CustomExceptions import NotUniqueException
from DataClassesJSON import ConfigData




class Data(ABC):
    def __init__(
        self, 
        config: ConfigData
    ) -> None:
        """Initialises an instance of Data where data is generated and processed.
        Sets random seeds for reproducability.

        INPUTS
        ------
            ``config`` is the configuration file of the models of type ConfigData.
        """
        self.train_size = config.train_size
        self.test_size = config.test_size
        self.degree = config.degree
        self.mean = config.mean
        self.std = config.std
        
        # for reproduction
        np.random.seed(42)
        torch.manual_seed(1234)


    # add noise to input array
    def addNoise(
        self, 
        input: np.ndarray
    ) -> np.ndarray:
        """Adds gaussian noise to the given input array with 
        the mean and the standard deviation given in config.

        INPUTS
        ------
            ``input`` is an array with the true datapoints.


        RETURN
        ------
            ``input_noise`` is an array with noisy datapoints.
        """
        input_noise = input + np.random.randn(*input.shape) * self.std + self.mean
        return input_noise


    # checks if the array is completely unique 
    # return true if its unique
    def inputUnique(
        self, 
        input: np.ndarray
    ) -> bool:
        """Checks if input datapoints are unique.

        INPUTS
        ------
            ``input``  is an array with all datapoints that are checked  
            (for 2d of shape (:, 1), for 3d of shape (:, 2),...).

        RETURN
        ------
            ``is_unique`` is a true if there are no duplicate datapoints. 
        """
        unique_input = np.unique(input, axis=0)
        dupl_len = len(input) - len(unique_input)
        is_unique = dupl_len==0
        if is_unique:
            return is_unique
        else:
            raise NotUniqueException(dupl_len)
        

    
    def getDataLoader(
        self, 
        data: tuple[np.ndarray,...], 
        batch_size: int, 
        loader_reprod: bool=True, 
        run: int = 0
    ) -> DataLoader:
        """Gets the dataloader with a configured batch size. 
        The data are stored as tensors and in device.

        INPUTS
        ------
            ``data`` is the data thats stored. 

            ``batch_size`` is the size of one batch in the loader. 
            
            ``loader_reprod`` define if the loader is manual seeded or not. 
            It is set to true (every loader has the same batches).

        RETURN
        ------
            ``dataloader`` is the generated dataloader. 
        """
        # def worker_init_fn(worker_id, run):
        #     torch.random.manual_seed(run)

        gen_loader = torch.Generator().manual_seed(187) if loader_reprod else torch.Generator().manual_seed(run)
        if (len(data[0]) % batch_size)==1:
            drop_last_batch = True
            print('Droped last batch beacause it is of size 1')
        else:
            drop_last_batch = False

        dataset = TensorDataset(*data)
        # num_workers = self.checkBestWorkers(dataset, batch_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            generator=gen_loader,
            drop_last=drop_last_batch,
            # worker_init_fn=lambda worker_id : worker_init_fn(worker_id, 187 if loader_reprod else run)
        )
        return dataloader


    def toTorch(
        self, 
        data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Modify data to tensor and store it on the chosen device.

        INPUTS
        ------
            ``data`` is the data as a tuple of two arrays with first the model input 
            and second the true output.

        RETURN
        ------
            ``input_tensor`` is a matrix stored as a tensor on chosen device. 

            ``output_tensor`` is a vector stored as a tensor on chosen device. 
        """
        input_np, output_np = data
        input_tensor = torch.from_numpy(input_np).float()
        output_tensor = torch.from_numpy(output_np).float()
        if Helpers.use_cuda:
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()
        return input_tensor, output_tensor


    def toNumpy(
        self, 
        data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray]:
        input_tensor, output_tensor = data
        input_np = input_tensor.detach().cpu().numpy()
        output_np = output_tensor.detach().cpu().numpy()
        return input_np, output_np

    
    def checkBestWorkers(self, dataset, batch_size):
        worker_list = list(range(0, mp.cpu_count(), 2))
        worker_perf = np.zeros((len(worker_list), 2))

        for i, num_workers in enumerate(worker_list):  
            train_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers
            )
            print(f'{num_workers} worker...')
            start = perf_counter()
            for _ in range(3):
                for _ in enumerate(train_loader):
                    pass
            end = perf_counter()
            worker_perf[i, :] = [num_workers, end - start]
        min_index = worker_perf[:, 1].argmin()
        print(f'Best worker setup: {worker_perf[min_index, 0]} - {worker_perf[min_index, 1]}')
        return worker_perf[min_index, 0]


    @abstractmethod
    def _makeData(self):
        """Generates original, training, validation and test data. 
        Ceck train and test datapoints if they are unique. +
        Validation data is generated out of random test data points. 
        the original data is the unprocessed test data. 
        """
        pass


    @abstractmethod
    def _makeOriginalData(self):
        pass


    @abstractmethod
    def _dataFun(self):
        pass
