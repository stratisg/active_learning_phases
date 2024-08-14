"""
Configuration for the Ising model and its simulation.
"""
import os
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch.nn import MSELoss
from torch.optim import SGD
from ising_model import IsingModel
from utilities import calculate_absolute_magnetization


# General variables.
data_dir = "../data"
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# Model parameters.
ising_args = dict(lattice=np.array([10, 10], dtype=int), radius=1, 
                  external_field=0, boundary_conds="periodic", seed=1959)

model = IsingModel(**ising_args)
model_name = model.model_name
model.generate_site_indices()
model.get_neighborhood()
site_indices = model.site_indices
data_dir_model = f"{data_dir}/{model.model_name}"
if not os.path.isdir(data_dir_model):
    os.mkdir(data_dir_model)

# Simulation configuration.
n_samples = int(1e5)
k_boltzmann = 1
warmup_iter = int(1e3)
# Simulation configuration.
n_samples = int(1e3)
k_boltzmann = 1
warmup_iter = int(1e3)
temperature = 2.0
simulation_args = dict(n_samples=n_samples, temperature=temperature,
                        k_boltzmann=k_boltzmann, warmup_iter=warmup_iter,
                        seed_warmup=1821, seed_generation=1917,
                        verbose=True, data_dir=data_dir_model)

# Parameters used to generate training data.
# Temperature.
temp_min = 0.5
temp_max = 3.0
temp_delta = 5e-1
l_temperatures = np.arange(temp_max, temp_min, -temp_delta)

# Interaction strength.
interaction_min = -2.0
interaction_max = 2.5
interaction_delta = 5e-1
l_interactions = np.arange(interaction_min, interaction_max,
                           interaction_delta)

# TODO: Probably rename.
training_pars = [l_temperatures, l_interactions]

# Training parameters.
# Define the model.
# TODO: Avoid hardcoding
n_hidden = 10
fit_model = Sequential(Linear(in_features=2, out_features=n_hidden),
                       ReLU(),
                       Linear(in_features=n_hidden, out_features=1)
)
optimizer_fn = SGD
optimizer_args = dict(lr=1e-3, momentum=0, dampening=0, weight_decay=0,
                      nesterov=False)
optimizer = optimizer_fn(fit_model.parameters(), **optimizer_args)
n_epochs = int(1e2)
n_points = int(1e2)
loss_fn = MSELoss()
verbose = True
training_args = {"loss_fn": loss_fn, "optimizer": optimizer,
                 "n_epochs": n_epochs, "verbose": verbose}

# Optimization parameters.
step = 1e-2
grad_threshold = 1e4
n_iter = int(1e3)
choose_args = {
    "step": 1e-2, "grad_threshold": 1e4, "n_iter": int(1e3),
    "bounds":[[1, -3], [10, 3]]
    }

# Quantity of interest
quant_name = "avg_magnetization"
quant_fn = calculate_absolute_magnetization
quant_args = {}

# Parameters used to generate results.
results_dir = "../results"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
results_dir_model = f"{results_dir}/{model_name}"
if not os.path.isdir(results_dir_model):
    os.mkdir(results_dir_model)


# Model directory.
model_dir = "../models"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model_dir_model = f"{model_dir}/{model_name}"
if not os.path.isdir(model_dir_model):
    os.mkdir(model_dir_model)

# Visualization parameters.
dpi = 600
pics_dir = "../pics"
if not os.path.isdir(pics_dir):
    os.mkdir(pics_dir)
pics_dir_model = f"../pics/{model_name}"
figname =  f"{model_name}"
if not os.path.isdir(pics_dir_model):
    os.mkdir(pics_dir_model)
