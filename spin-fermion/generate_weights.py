import argparse
import time
import os
import numpy as np
from training_parameters import n_train, n_valid
from training_parameters import data_augm
from generate_samples_config import dimension, hopping, interaction
from generate_samples_config import chemical_potential
from generate_samples_config import n_samples, model_name, data_augm
from generate_samples_config import interaction_model
from importance_sampling import get_importance_sampling
from utilities import cart_to_sph
from utilities import get_hkin
from utilities import free_energy_exact


def write_weights_file():
    """Writes a file with the weighted average for each quantity, its
    uncertainty, and the temperature."""
    dirname = "docs/generate_curves/"
    if not os.path.isdir("docs"):
        os.mkdir("docs")
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    model_name = filename.split("/")[-1].split("_")[0]
    data_arr = np.load(filename)
    l_free_energy = data_arr["f"]
    eigen_energies = data_arr["evals"]
    l_spin_components = np.array([data_arr["x"], data_arr["y"],
                                  data_arr["z"]])
    if importance_sampling:
        geometry = n_spins * np.ones(dimension, dtype=np.int32)
        n_system = int(n_spins ** dimension)
        hamiltonian = np.zeros((2 * n_system, 2 * n_system),
                               dtype=np.complex128)
        hamiltonian_kin = get_hkin(geometry)
        hamiltonian_args = (geometry, hopping, interaction, chemical_potential,
                            hamiltonian, hamiltonian_kin.data,
                            hamiltonian_kin.indices, hamiltonian_kin.indptr)
        for i_sample in range(n_samples):
            thetas, phis = cart_to_sph(l_spin_components[0, i_sample],
                                       l_spin_components[1, i_sample],
                                       l_spin_components[2, i_sample])
            _, eigendecomp = free_energy_exact(thetas, phis, beta,
                                               *hamiltonian_args)
            eigen_energies[i_sample] = eigendecomp[0]
        weights, _ = get_importance_sampling(l_free_energy, eigen_energies,
                                             n_spins, dimension, beta,
                                             chemical_potential)
    else:
        weights = np.ones(n_samples)
    file_header = "weights_"
    if not importance_sampling:
        file_header += "noimportance_"
    weights_file = dirname + file_header + filename.split("/")[-1]
    np.savez(weights_file, weights=weights)


script_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("n_spins", type=int, help="Length of chain")
parser.add_argument("window", type=int, help="Size of scanning window")
parser.add_argument("beta", type=float, help="Inverse temperature")
parser.add_argument("importance_sampling", type=int, choices=[0, 1],
                    help="0: w/o importance sampling,"
                         "1: w/ importance sampling")
parser_args = parser.parse_args()
n_spins = parser_args.n_spins
window = parser_args.window
beta = parser_args.beta
importance_sampling = parser_args.importance_sampling
if model_name == "Exact":
    data_augm = ""    
else:
    data_augm += "_"
dir_name = f"data/{model_name}/"
filename = generate_filename(model_name, data_augm, n_spins, dimension,
                             hopping, interaction, n_samples, beta,
                             chemical_potential, window, n_train, n_valid,
                             interaction_model)
filename = dir_name + filename                      
if model_name == "Exact":
    filename = filename.split("_train")[0] + ".npz"
write_weights_file()
script_duration = int(time.time() - script_start)
print(f"Script duration: {script_duration} seconds!")

