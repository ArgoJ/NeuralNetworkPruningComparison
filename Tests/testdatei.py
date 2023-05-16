import torch
import pandas as pd
import os, sys
import numpy as np
import json
import pickle as pkl
import matplotlib.pyplot as plt


from time import time
from timeit import timeit



# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Saves import save_JSON, get_method_str
from Loads import getDimensionSaveDirectory
from Pruning.PrunHelpers import calc_iter_amount
from mainEvals import makeModelEvals
from Loads import Loadings, loadConfig
from Helpers import selectDevice, find_model, add_to_existing_frame_file
from CustomExceptions import DimensionException
from DataClasses import Models
from NeuralNetwork import evalModel
from DataClassesJSON import PrunConfig, MethodPrunConfig

from DataScripts.Data2d import Data2d
from DataScripts.Data3d import Data3d

from Illustrations.ModelDraws import makeDraws

from Pruning.PrunHelpers import getPrunConfig, removeNodes


if __name__ == '__main__':
    insert_path = input('Type in the additional path!\n').replace('"', '')
    insert_df: pd.DataFrame = pd.read_pickle(insert_path)
    
    print(insert_df.index.droplevel(level=(7, 8)).unique().size)

    print()