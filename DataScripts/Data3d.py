import torch
import numpy as np


import MetricFunctions
import NeuralNetwork
from NeuralNetwork import Model
from DataClassesJSON import ConfigData

from DataScripts.Data import Data




class Data3d(Data):
    def __init__(
        self, 
        config: ConfigData
    ) -> None:
        """Initialises an instance of Data3d where 3d data is generated and processed.

        INPUTS
        ------
        ``config`` is the configuration file of the models of type ConfigData.
        """
        super().__init__(config)
        self._makeData()


    def _dataFun(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> np.ndarray:
        """Calculates z as a function of x and y.  
        z = 42*np.tanh(y/24) - 7*np.cos((y**2 + x**2)/842) + 19*np.exp(-x/24)*np.cos((x+y)/15)

        INPUTS
        ------
            ``x`` and ``y`` are arrays with the same row size as column size.

        RETURN
        ------
            ``out`` is an array with the values z as a function of x and y.
        """
        out = 42*np.tanh(y/24) - 7*np.cos((y**2 + x**2)/842) + 19*np.exp(-x/24)*np.cos((x+y)/15)
        return out
    

    def _makeData(self, print_datasizes = False):
        """Generates training, validation and test data in the square of 100*100. 
        Ceck train and test datapoints if they are unique. 

            ``original_data`` is the unprocessed test data (linspace mesh). 

            ``train_data`` are random generated datapoints processed for the model. 

            ``val_tensor`` are 20% of random test data points. 

            ``test_data`` are datapoints generated with linspace and mesh. 
        """
        # train, validation and testdata generation
        x_train, y_train, z_train = self._getRandTrainData()
        x_test, y_test, z_test = self._makeOriginalData()

        # check if datapoints of train and test data are completely unique
        self._checkUniqueness(x_train, y_train, x_test, y_test)

        self.original_data = x_test, y_test, z_test
        train_data = self._reshapeForModel((x_train, y_train, self.addNoise(z_train)))
        test_data = self._reshapeForModel((x_test, y_test, z_test))

        self.train_tensor = self.toTorch(train_data)
        self.test_tensor = self.toTorch(test_data)

        # take 20% of random testdata points for validation data
        index = np.random.choice(self.test_size**2, int(0.2*(self.test_size**2)), replace=False)
        self.val_tensor = self.test_tensor[0][index], self.test_tensor[1][index]

        # print sizes of arrays
        if print_datasizes:
            print(f'Traindata size: {x_train.shape}')
            print(f'Validationdata size: {self.val_tensor[0].shape[0]}')
            print(f'Testdata size: {x_test.shape}')



    def _getRandTrainData(self) -> tuple[np.ndarray,...]:
        """Creates random datapoints in a square in the z plane of 100*100.  

        RETURN
        ------
            ``x``, ``y`` are arrays with random generated datapoints and 

            ``z`` is an array z as a function of x and y.
        """
        x = np.random.rand(self.train_size, self.train_size)*100
        y = np.random.rand(self.train_size, self.train_size)*100
        z = self._dataFun(x, y)
        return x, y, z


    def _makeOriginalData(self) -> tuple[np.ndarray,...]:
        """Creates linspace datapoints in a square in the z plane of 100*100. 

        RETURN
        ------
            ``x``, ``y`` are linspace generated datapoints and 
            ``z`` is an array z as a function of x and y.
        """
        x, y = self._getLinspaceMesh(self.test_size)
        z = self._dataFun(x, y)
        return x, y, z


    def getModelData(
        self, 
        model: Model, 
        true_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process the original data for the model and then get the predictions of the model. 

        INPUTS
        ------
            ``model`` is the model that predicts the values. 

            ``true_data`` are the unprocessed test data (just need x and y).

        RETURN
        ------
            ``x``, ``y`` are unprocessed linspace generated datapoints and 

            ``z`` is an unprocessed array with the predictions of the model.
        """
        x, y, _ = true_data
        x_vect = self._matrixToVector(x)
        y_vect = self._matrixToVector(y) 
        xy_vect = self._reshapeToXY((x_vect, y_vect))
        z_vec = NeuralNetwork.getModelPreds(model, xy_vect)
        z = self._vectorToMatrix(z_vec)
        return x, y, z


    def _getLinspaceMesh(
        self, 
        sq_point: int
    ) -> tuple[np.ndarray,...]:
        """Generates a mesh of linspace datapoints. 

        INPUTS
        ------
            ``sq_point`` is the size of the mesh generated (row and column size).

        RETURN
        ------
            ``x`` and ``y`` is a linspace generated mesh.
        """
        x = np.linspace(0, 100, sq_point)
        x, y = np.meshgrid(x, x)
        return x, y


    def getError(
        self, 
        model: Model, 
        true_data: tuple[np.ndarray,...]
    ) -> tuple[np.ndarray,...]:
        """Get the error data as the error of model prediction and z of the test data.

        INPUTS
        ------
            ``model`` is the model that predicts the values. 

            ``true_data`` are the unprocessed test datapoints. 

        RETURN
        ------
            ``error_data`` are the test datapoints with 
            the associated error of model prediction and z of the test data.
        """
        model_data = self.getModelData(model, true_data)
        model_pred = model_data[2]
        error = MetricFunctions.calcError(model_pred, true_data[2])
        return model_data[0], model_data[1], error


    def _reshapeForModel(
        self, 
        data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reshapes the input x and y for the model to the shape of (:, 2*degree)

        INPUTS
        ------
            ``data`` is the data as a tuple of three arrays.

        RETURN
        ------
            ``xy`` are the datapoints put together in one array (shape of (:, 2*degree)). 

            ``z`` is the z array not changed.
        """
        x, y, z = self._reshapeDataToVector(data)
        xy = self._reshapeToXY((x, y))
        return xy, z

    
    def reshapeForPlot(
        self, 
        data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reshapes the input data from model processed data to plot ready unprocessed data.

        INPUTS
        ------
            ``data`` is the data as a tuple of two arrays with data processed for model 
            (first xy and second z).

        RETURN
        ------
            ``x_re``, ``y_re`` and ``z_re`` are matrices reshaped from vecotrs 
            and degrees are removed. 
        """
        xy, z = data
        xy, z = xy.detach().cpu().numpy(), z.detach().cpu().numpy()
        x = xy[:, 0]
        y = xy[:, self.degree]
        x_re = self._vectorToMatrix(x)
        y_re = self._vectorToMatrix(y)
        z_re = self._vectorToMatrix(z)
        return x_re, y_re, z_re


    def _reshapeDataToVector(
        self, 
        data: tuple[np.ndarray,...]
    ) -> tuple[np.ndarray,...]:
        """Reshapes the input data from vectors to matrices.

        INPUTS
        ------
            ``data`` is the data as a tuple of three vectors. 

        RETURN
        ------
            ``x_re``, ``y_re`` and ``z_re`` are matrices reshaped from vectors.
        """
        x, y, z = data
        x_re = self._matrixToVector(x)
        y_re = self._matrixToVector(y)
        z_re = self._matrixToVector(z)
        return x_re, y_re, z_re


    def _matrixToVector(
        self, 
        input: np.ndarray
    ) -> np.ndarray:
        """Reshapes the input matrix to a vector of shape (:, 1).

        INPUTS
        ------
            ``input`` is a matrix.

        RETURN
        ------
            ``out`` is a vector of shape (:, 1).
        """
        return input.reshape(-1, 1)


    def _reshapeToXY(
        self, 
        xy_tuple: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Reshape the vectors x and y to one array made model ready. 

        INPUTS
        ------
            ``xy_tuple`` are the seperate vectors x and y. 

        RETURN
        ------
            ``xy_poly`` is one xy array with its degrees (e.g. (x, x², y, y²) if degree = 2).
        """
        x, y = xy_tuple
        x_poly = [x ** i for i in range(1, self.degree + 1)]
        y_poly = [y ** i for i in range(1, self.degree + 1)]
        poly_tuple = x_poly + y_poly
        return np.concatenate(poly_tuple, axis=1)


    def _vectorToMatrix(
        self, 
        vector: np.ndarray
    ) -> np.ndarray:
        """Reshape a matrix to vector with same row and column length. 

        INPUTS
        ------
            ``vector`` is the input vector of shape (:, 1).

        RETURN
        ------
            ``matrix`` is the output matrix with same row and column length. 
        """
        m_size = np.int_(np.sqrt(len(vector)))
        return vector.reshape((m_size, m_size))

    
    def _checkUniqueness(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray
    ) -> bool:
        """Concentenate the x's and y's into xy vector array of shape (:, 2) each. 
        Combines xy1 and xy2 and check if the datapoints are unique.

        INPUTS
        ------
            ``x_train`` and ``y_train`` are the unprocessed train datapoints

            ``x_test`` and ``y_test`` are the unprocessed test datapoints

        RETURN
        ------
            ``is_unique`` is a true if there are no duplicate datapoints. 
        """
        xy_train = np.concatenate((self._matrixToVector(x_train), self._matrixToVector(y_train)), axis=1)
        xy_test = np.concatenate((self._matrixToVector(x_test), self._matrixToVector(y_test)), axis=1)
        xy = np.concatenate((xy_train, xy_test))
        return self.inputUnique(xy)

    
    def getTrueDataSize(self) -> tuple[int]:
        """Gets the true datapoint sizes of test and train.

        RETURN
        ------
            ``true_train_size``, ``true_test_size`` 
        """
        return self.train_size**2, self.test_size**2


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