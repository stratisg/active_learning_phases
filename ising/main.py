"""
Training/Fit model given a dataset
"""
import os
import glob
import numpy as np
import torch
from config import n_epochs, n_points, loss_fn, fit_model, optimizer
from config import model, simulation_args
from config import data_dir_model, results_dir_model
from simulation import Simulation
from training import training
from analyze_samples import get_quantity
from utilities import calculate_absolute_magnetization, load_quantity

# 1. Load initial training set.
quant_name = "avg_magnetization"
l_values = load_quantity(results_dir_model, quant_name)
l_quantity_mean = np.zeros(len(l_values), dtype=float)
l_temperatures = np.zeros_like(l_quantity_mean)
l_interactions = np.zeros_like(l_quantity_mean)
for i_val, d_vals in enumerate(l_values):
    l_quantity_mean[i_val] = d_vals["quantity_mean"]
    l_temperatures[i_val] = d_vals["temperature"]
    l_interactions[i_val] = d_vals["interaction"]
l_data = torch.tensor([l_temperatures, l_interactions, l_quantity_mean],
                      dtype=torch.float).T

for i_pt in range(n_points):
    # 2. Train/fit the model.
    model = training(l_data, fit_model, loss_fn, optimizer, n_epochs)
    quit()
    # 3. Suggest a new set of parameters to simulate.
    # We find the point that maximizes the magnitude of the gradient of
    # the model with respect to its input.
    # TODO: Under construction.
    # temperature, interaction = 
    # 4. Run the simulation.
    model.interaction = interaction
    simulation = Simulation(model, **simulation_args)
    simulation.generate_samples()

    # 5. Analyze the data of the simulation.
    filename = f"{data_dir_model}_temperature_{temperature}_" \
        f"interaction_{interaction}"
    quant_fn = calculate_absolute_magnetization
    quant_args = {}
    get_quantity(filename, quant_name, quant_fn, quant_args,
                 results_dir_model)
# 6. Append the results of the simulation on the training set.
# 7. Repeat from step 2 until a certain criterion is satisfied. 