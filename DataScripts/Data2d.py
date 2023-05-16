import numpy as np


import NeuralNetwork
from NeuralNetwork import Model
from DataClassesJSON import ConfigData

from DataScripts.Data import Data



class Data2d(Data):
    def __init__(
        self, 
        config: ConfigData
    ) -> None:
        """Initialises an instance of Data2d where 2d data is generated and processed.

        INPUTS
        ------
            ``config`` is the configuration file of the models of type ConfigData.
        """
        super().__init__(config)
        self._makeData()


    def _dataFun(
        self, 
        x: np.ndarray
    ) -> np.ndarray:
        """Calculates y as a function of x.  
        z = np.exp(-x/21)*(56*np.cos(2*x/6) + 19*np.sin((x+3)/2)) + 64*np.tanh(x/24) + 2*np.sin(x/3)

        INPUTS
        ------
            ``x`` is a vector.

        RETURN
        ------
            ``out`` is an vector with the values y as a function of x.
        """
        out = np.exp(-x/21)*(56*np.cos(2*x/6) + 19*np.sin((x+3)/2)) + 64*np.tanh(x/24) + 2*np.sin(x/3)
        return out


    def _makeData(self, print_datasizes = False):
        """Generates training, validation and test data in the range of 0 to 100. 
        Ceck train and test datapoints if they are unique. 

            ``original_data`` is the unprocessed test data (linspace). 

            ``train_data`` are random generated datapoints processed for the model. 

            ``val_tensor`` are 50% of random test data points. 
            
            ``test_data`` are datapoints generated with linspace. 
        """
        # rand train generation
        x_train, y_train = self.getRandomData(self.train_size)
        # test data
        x_test, y_test = self._makeOriginalData()

        # check if train and test is completaly unique
        self._checkUniqueness(x_train, x_test)

        self.original_data = x_test, y_test
        train_data = self._reshapeForModel(x_train), self.addNoise(y_train)
        test_data = self._reshapeForModel(x_test), y_test

        self.train_tensor = self.toTorch(train_data)
        self.test_tensor = self.toTorch(test_data)

        # take 50% of random testdata points for validation data
        index = np.random.choice(self.test_size, int(0.5*self.test_size), replace=False)
        self.val_tensor = self.test_tensor[0][index], self.test_tensor[1][index]

        # print sizes
        if print_datasizes:
            print(f"Total data size: {self.train_size + self.test_size}")
            print(f"Traindata size: {len(x_train)}")
            print(f"Validationdata size: {len(self.val_tensor[0])}")
            print(f"Testdata size: {len(x_test)}")


    def getRandomData(
        self, 
        data_size: int
    ) -> tuple[np.ndarray,...]:
        """Creates random datapoints in the range of 0 to 100.  

        RETURN
        ------
            ``x`` is a vector with random generated datapoints and 

            ``y`` is a vector y as a function of x.
        """
        x = np.random.rand(data_size, 1)*100
        y = self._dataFun(x)
        return x, y


    def _makeOriginalData(self) -> tuple[np.ndarray,...]:
        """Creates linspace datapoints in the range of 0 to 100.  

        RETURN
        ------
            ``x`` is a linspace generated vector and 

            ``y`` is a vector y as a function of x.
        """
        x = getLinspace(self.test_size)
        y = self._dataFun(x)
        return x, y


    def getModelData(
        self, 
        model: Model, 
        data: tuple[np.ndarray,...]
    ) -> tuple[np.ndarray,...]:
        """Process the original data for the model and then get the predictions of the model. 

        INPUTS
        ------
            ``model`` is the model that predicts the values. 

            ``true_data`` are the unprocessed test data (just need x).

        RETURN
        ------
            ``x`` is the unprocessed linspace generated data and 

            ``y`` is an unprocessed vector with the predictions of the model .
        """
        x, _ = data
        x_poly = self._reshapeForModel(x)
        y = NeuralNetwork.getModelPreds(model, x_poly)
        return x, y


    def _reshapeForModel(
        self, 
        input: np.ndarray
    ) -> np.ndarray:
        """Reshapes the input vector for the model to the shape of (:, degree)

        INPUTS
        ------
            ``input`` is a unprocessed vector.

        RETURN
        ------
            ``input_poly`` is an array with (x, x², x³,..., x^n) degree=n
        """
        input_poly = np.concatenate([input ** i for i in range(1, self.degree + 1)], 1)
        return input_poly


    def _reshapeForPlot(
        self, 
        data: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reshapes the input data from model processed data to plot ready unprocessed data.

        INPUTS
        ------
            ``data`` is the data as a tuple of two arrays with data processed for model 
            (first x and second y).

        RETURN
        ------
            ``x_re`` is a vector with the degree removed. 

            ``y_re`` is unprocessed.
        """
        x, y = data
        return x[:, 0], y

    
    def _checkUniqueness(
        self, 
        x_train: np.ndarray, 
        x_test: np.ndarray 
    ):
        """Combines x's and check if the datapoints are unique.

        INPUTS
        ------
            ``x_train`` is the unprocessed train vector. 

            ``x_test`` is the unprocessed train vector.

        RETURN
        ------
            ``is_unique`` is a true if there are no duplicate datapoints. 
        """
        x = np.concatenate((x_train, x_test))
        self.inputUnique(x)
    

    def getTrueDataSize(self) -> tuple[int]:
        """Gets the true datapoint sizes of test and train.

        RETURN
        ------
            ``true_train_size``, ``true_test_size`` 
        """
        return self.train_size, self.test_size


    def getTrainData(self):
        """Gets the generated training data. 

        RETURN
        ------
            ``train_data`` is random generated test data. 
        """
        return self.train_tensor


    def getTestData(self):
        """Gets the generated test data. 

        RETURN
        ------
            ``test_data`` is with a linspace generated test data. 
        """
        return self.test_tensor

    
    def getValidationTensor(self):
        """Gets random points of the generated test data as 
        ``validation_tensor``. 

        RETURN
        ------
            ``validation_tensor`` are points of the generated test data. 
        """
        return self.val_tensor


    def getOriginalData(self):
        """Gets the original linspace data unprocessed for the model.  

        RETURN
        ------
            ``original_data`` is the unprocessed linspace generated data. 
        """
        return self.original_data


def getLinspace(points: int) -> np.ndarray:
    """Generates a mesh of linspace datapoints. 

    INPUTS
    ------
    ``sq_point`` is the size of the mesh generated (row and column size).

    RETURN
    ------
    ``x`` and ``y`` is a linspace generated mesh.
    """
    linsp = np.linspace(0, 100, points)
    linsp = linsp.reshape((len(linsp), 1))
    return linsp