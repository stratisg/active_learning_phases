import os
import numpy as np
from simulation import Simulation
from ising_model import IsingModel
from ising_configuration import ising_args, n_samples, k_boltzmann, warmup_iter


if not os.path.isdir("../data"):
    os.mkdir("../data")

ising_model = IsingModel(**ising_args)
ising_model.generate_site_indices()
ising_model.get_neighborhood()

# Generate data to fit the model used to identify the boundary 
# between phases. Build a grid with different values for temperature 
# and interaction strength.
# TODO: Potentially use Latin hypercube for building the training set.

# Temperature.
temp_min = 0.5
temp_max = 3.0
temp_delta = 5e-1
l_temperatures = np.arange(temp_max, temp_min, -temp_delta)

# Interaction strength.
interaction_min = -2.0
interaction_max = 2.5
interaction_delta = 5e-1
l_interactions = np.arange(interaction_min, interaction_max, interaction_delta)

for temperature in l_temperatures:
    print(79 * "=")
    print(f"Temperature {temperature:.3f}")
    for interaction in l_interactions:
        if not interaction:
            continue
        print(39 * "=")
        print(f"Interaction {interaction:.3f}")
        ising_model.interaction = interaction
        simulation_args = dict(n_samples=n_samples, temperature=temperature,
                            k_boltzmann=k_boltzmann,
                            warmup_iter=warmup_iter, seed_warmup=1821,
                            seed_generation=1917, verbose=True,
                            data_dir="../data/ising")
        simulation = Simulation(ising_model, **simulation_args)
        simulation.generate_samples()
    