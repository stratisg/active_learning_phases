"""
Training/Fit model given a dataset
"""
import numpy as np
import torch
from config import n_points, fit_model, training_args
from config import model, simulation_args
from config import data_dir_model, results_dir_model
from simulation import Simulation
from training import training
from optimize import optimize
from analyze_samples import save_quantity
from utilities import calculate_absolute_magnetization, load_data


# 1. Load initial training set.
quant_name = "avg_magnetization"
l_values = load_data(results_dir_model, quant_name)
l_quantity_mean = np.zeros(len(l_values), dtype=float)
l_temperatures = np.zeros_like(l_quantity_mean)
l_interactions = np.zeros_like(l_quantity_mean)
for i_val, d_vals in enumerate(l_values):
    l_quantity_mean[i_val] = d_vals["quantity_mean"]
    l_temperatures[i_val] = d_vals["temperature"]
    l_interactions[i_val] = d_vals["interaction"]
l_data_in = np.array([l_temperatures, l_interactions])
l_data_in = torch.tensor(l_data_in.T, dtype=torch.float)
l_data_out = torch.tensor(l_quantity_mean, dtype=torch.float)
l_data_out = torch.reshape(l_data_out, (len(l_data_in), 1))

for i_pt in range(n_points):
    print(79 * "=")
    print(f"Test point {i_pt:03d}")
    
    # 2. Train/fit the model.
    fit_model = training(l_data_in, l_data_out, fit_model, **training_args)

    # 3. Suggest a new set of parameters to simulate.
    # We find the point that maximizes the magnitude of the gradient of
    # the model with respect to its input.
    
    # TODO: Find the set of points that satisfy a certain objective
    # such as we have the highest uncertainty or most sensitivity to
    # input parameters.
    # TODO: Put bounds for temperature so it stays positive.
    # TODO: Put bounds for parameters.
    temperature, interaction = optimize(fit_model, n_input=l_data_in.shape[1])
    print(39 * "-")
    print("Optimized parameters")
    print(f"temperature {temperature}")
    print(f"interaction {interaction}")
    print(39 * "-")

    # 4. Run the simulation.
    simulation_args["temperature"] = temperature
    model.interaction = interaction
    simulation = Simulation(model, **simulation_args)
    simulation.generate_samples()

    # 5. Save the data and calculate the order parameter of the simulation.
    filename = f"{data_dir_model}/data_{model.model_name}_" \
               f"temperature_{temperature}_interaction_{interaction}.npz"
    quant_fn = calculate_absolute_magnetization
    quant_args = {}
    quantity_mean, quantity_std = save_quantity(filename, quant_name, quant_fn,
                                               quant_args, results_dir_model)
    
    # 6. Append the results of the simulation on the training set.
    data_in_new = torch.tensor([[temperature, interaction]], dtype=torch.float)
    l_data_in = torch.cat([l_data_in, data_in_new])
    data_out_new = torch.tensor([[quantity_mean]], dtype=torch.float)
    l_data_out = torch.cat([l_data_out, data_out_new])

    # 7. Repeat from step 2 until a certain criterion is satisfied.

# TODO: Generalize to a generic model parameters and order parameter.