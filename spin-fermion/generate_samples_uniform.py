import time
import argparse
import os
import numpy as np
from generate_samples_config import dimension, n_spins
from generate_samples_config import hopping, interaction
from generate_samples_config import n_samples
from utilities import lattice_geometry
from utilities import sph_to_cart
from utilities import construct_hamiltonian


script_start = int(time.time())
random_gen = np.random.default_rng()
geometry = lattice_geometry(n_spins, dimension)
n_system = n_spins ** dimension
x_samples = np.zeros([n_samples, n_system])
y_samples = np.zeros([n_samples, n_system])
z_samples = np.zeros([n_samples, n_system])
evals = np.zeros([n_samples, 2 * n_system])
evecs = np.zeros([n_samples, 2 * n_system, 2 * n_system], dtype=np.complex128)
if not os.path.isdir("data"):
    os.mkdir("data")
data_dir = "data/ExactUniform"
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
for i_sample in range(n_samples):
    thetas = np.arccos(random_gen.uniform(-1, 1, size=n_system))
    phis = random_gen.uniform(-np.pi, np.pi, size=n_system)
    x_, y_, z_ = sph_to_cart(thetas, phis)
    x_samples[i_sample] = x_
    y_samples[i_sample] = y_
    z_samples[i_sample] = z_
    hamiltonian = construct_hamiltonian(thetas, phis, geometry, hopping,
                                        interaction)
    evals[i_sample], evecs[i_sample] = np.linalg.eigh(hamiltonian)
filename = f"{data_dir}/ExactUniform_sites_{n_system}_dimension_" \
           f"{dimension}_hopping_{hopping}_interaction_{interaction}_" \
           f"samples_{n_samples}.npz"
np.savez(filename, x=x_samples, y=y_samples, z=z_samples, evals=evals,
         evecs=evecs)
script_duration = int(time.time()) - script_start
print(f"Script duration: {script_duration} seconds!")

