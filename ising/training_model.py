"""
Training/Fit model given a dataset
"""
import os
import glob
import numpy as np
import torch
from config import data_dir


# Load data.
l_train_files = glob.glob(f"{data_dir}/{model_name}")
np.load()
# 