"""
Configuration for the Ising model and its simulation.
"""
import os
import numpy as np
from ising_model import IsingModel


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
data_dir_model = f"{data_dir}/{model.model_name}"

# Simulation configuration.
n_samples = int(1e5)
k_boltzmann = 1
warmup_iter = int(1e3)

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

training_pars = [l_temperatures, l_interactions]

# Visualization parameters.
dpi = 600
pics_dir = "../pics"
if not os.path.isdir(pics_dir):
    os.mkdir(pics_dir)
pics_dir_model = f"../pics/{model_name}"
