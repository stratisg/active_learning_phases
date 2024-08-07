"""
Training/Fit model given a dataset
"""
import os
import glob
import numpy as np
import torch
from config import model, simulation_args
from config import data_dir_model, results_dir_model
from simulation import Simulation
from analyze_samples import get_quantity
from utilities import calculate_absolute_magnetization

# Load data.
quant_name = "avg_magnetization"
l_train_files = glob.glob(f"{results_dir_model}_{quant_name}*.npz")
d_pars = 

# 1. Load initial training set.
data_train = torch.zeros
for i_epoch in range(n_epochs):
    print(79 * "=")
    print(f"Epoch {i_epoch}")
    # 2. Train/fit the model.
    
    # 3. Suggest a new set of parameters to simulate.
    
    # 4. Run the simulation.
    model.interaction = interaction
    simulation = Simulation(model, **simulation_args)
    simulation.generate_samples()

    # 5. Analyze the data of the simulation.
    filename = f"{data_dir_model}_temperature_{temperature}_" \
        f"interaction_{interaction}"
    quant_fn = calculate_absolute_magnetization
    quant_args = {}
    get_quantity(filename, quant_name, quant_fn, quant_args, results_dir_model)
    # 6. Append the results of the simulation on the training set.
    # 7. Repeat from step 2 until a certain criterion is satisfied. 