import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from optuna.visualization import plot_parallel_coordinate
from plotly.io import show
import optuna

import data_utils

def use_NN_model(model: nn.Module, data_x: np.array) -> np.array:
    #TODO
    return None

