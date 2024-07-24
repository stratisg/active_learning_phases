import argparse
import time
import os
import numpy as np
from utilities import generate_filename
from thermodynamic_quantities import get_average_energy
from thermodynamic_quantities import get_specific_heat
from thermodynamic_quantities import get_abs_magnetization
from thermodynamic_quantities import get_stagg_magnetization
from training_parameters import n_train, n_valid
from training_parameters import data_augm
from generate_samples_config import dimension, hopping, interaction
from generate_samples_config import chemical_potential
from generate_samples_config import n_samples, model_name, data_augm
from generate_samples_config import interaction_model


def write_quantities_file():
    """Writes a file with the weighted average for each quantity, its
    uncertainty, and the temperature."""
    if not os.path.isdir("docs"):
        os.mkdir("docs")
    quantities_dir = "docs/generate_curves/"
    if not os.path.isdir(quantities_dir):
        os.mkdir(quantities_dir)
    l_quantities = np.zeros(4)
    l_errors = np.zeros_like(l_quantities)
    data_arr = np.load(datafile)
    eigen_energies = data_arr["evals"]
    l_spin_components = np.array([data_arr["x"], data_arr["y"], data_arr["z"]])
    if importance_sampling and model_name == "N1Eigenvalues":
        weights_file = quantities_dir + "weights_" + filehead
        d_weights = np.load(weights_file)
        weights = d_weights["weights"]
    else:
        weights = np.ones(n_samples) / n_samples
    avg_energy_ = get_average_energy(eigen_energies, weights, beta,
                                     chemical_potential)
    spec_heat_ = get_specific_heat(eigen_energies, weights, beta,
                                   chemical_potential)
    abs_magn = get_abs_magnetization(l_spin_components, weights)
    stagg_magn = get_stagg_magnetization(l_spin_components, n_spins, dimension,
                                         weights)
    l_quantities[0] = avg_energy_[0]
    l_errors[0] = avg_energy_[1]
    l_quantities[1] = spec_heat_[0]
    l_errors[1] = spec_heat_[1]
    l_quantities[2] = abs_magn[0]
    l_errors[2] = abs_magn[1]
    l_quantities[3] = stagg_magn[0]
    l_errors[3] = stagg_magn[1]
    quant_header = "quantities_"
    if not importance_sampling:
        quant_header += "noimportance_"
    quantities_file = quantities_dir + quant_header + filehead
    np.savez(quantities_file, quantities=l_quantities, errors=l_errors,
             weights=weights)


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
beta = parser_args.beta
window = parser_args.window
importance_sampling = parser_args.importance_sampling
if model_name == "Exact":
    data_augm = ""
filehead = generate_filename(model_name, data_augm, n_spins, dimension,
                             hopping, interaction, n_samples, beta,
                             chemical_potential, window, n_train, n_valid,
                             interaction_model)
datafile = f"data/{model_name}/{filehead}"
if model_name == "Exact":
    datafile = datafile.split("_train")[0] + ".npz"
print(f"datafile {datafile}")
write_quantities_file()
script_duration = int(time.time() - script_start)
print(f"Script duration: {script_duration} seconds!")

