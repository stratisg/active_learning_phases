"""
Configuration for the Ising model and its simulation.
"""
import numpy as np


# Model parameters.
ising_args = dict(lattice=np.array([10, 10], dtype=int), radius=1, 
                  external_field=0, boundary_conds="periodic", seed=1959)


# Simulation configuration.
n_samples = int(1e5)
k_boltzmann = 1
warmup_iter = int(1e3)
