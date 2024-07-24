import argparse
import time
import os
import numpy as np
from utilities import generate_filename
from training_parameters import n_train, n_valid
from training_parameters import data_augm
from generate_samples_config import dimension, hopping, interaction
from generate_samples_config import chemical_potential
from generate_samples_config import n_samples, model_name, data_augm
from generate_samples_config import interaction_model
from thermodynamic_quantities import get_spin_correlation


def write_spin_correlation_file():
    """Writes a file with the weighted average for each quantity, its
    uncertainty, and the temperature."""
    if not os.path.isdir("docs"):
        os.mkdir("docs")
    dirname = "docs/generate_curves/"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    data_arr = np.load(datafile)
    eigen_energies = data_arr["evals"]
    l_spin_components = np.array([data_arr["x"], data_arr["y"],
                                  data_arr["z"]])
    if importance_sampling and model_name == "N1Eigenvalues":
        weights_file = dirname + "weights_" + filehead
        d_weights = np.load(weights_file)
        weights = d_weights["weights"]
    else:
        weights = np.ones(n_samples) / n_samples
    l_spin_corr = get_spin_correlation(l_spin_components, n_spins, dimension,
                                       weights)
    spin_corr_head = "spin_correlation_"
    if not importance_sampling:
        spin_corr_head += "noimportance_"
    spin_corr_file = dirname + spin_corr_head + filehead
    np.savez(spin_corr_file, weights=weights, spin_corr=l_spin_corr[0],
             spin_corr_errors=l_spin_corr[1], r_vector=l_spin_corr[2])


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
filehead = generate_filename(model_name, data_augm, n_spins, dimension,
                             hopping, interaction, n_samples, beta,
                             chemical_potential, window, n_train, n_valid,
                             interaction_model)
datafile = f"data/{model_name}/{filehead}"
if model_name == "Exact":
    datafile = datafile.split("_train")[0] + ".npz"
print(f"datafile {datafile}")
write_spin_correlation_file()
script_duration = int(time.time() - script_start)
print(f"Script duration: {script_duration} seconds!")

