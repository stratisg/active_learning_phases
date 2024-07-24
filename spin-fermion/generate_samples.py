import os
import glob
import argparse
import time
import numpy as np
import torch
from training_parameters import n_train, n_valid
from training_parameters import hidden_1, loss
from training_parameters import translation_toggle, rotation_toggle
from training_parameters import model_toggle
from generate_samples_config import dimension, hopping, interaction
from generate_samples_config import chemical_potential
from generate_samples_config import n_samples, model_name, data_augm
from generate_samples_config import scanning_stride
from generate_samples_config import n_checkpoint, n_warmup
from generate_samples_config import interaction_model
from utilities import get_hkin
from utilities import free_energy_exact
from utilities import sample_from_evals
from utilities import sph_to_cart
from utilities import sample_from_window
from metropolis import metropolis, metropolis_adapt


script_start = int(time.time())
parser = argparse.ArgumentParser()
parser.add_argument("n_spins", type=int, help="Length of chain")
parser.add_argument("window", type=int, help="Size of scanning window")
parser.add_argument("beta", type=float, help="Inverse temperature")
parser.add_argument("--metro_adapt", type=int, choices=[0, 1], default=0,
                    help="Use of prior for metropolis")
parser_args = parser.parse_args()
n_spins = parser_args.n_spins
window = parser_args.window
beta = parser_args.beta
metro_adapt = parser_args.metro_adapt
pars = {}
pars["n_spins"] = n_spins
pars["beta"] = beta
pars["n_train"] = n_train
pars["n_samples"] = n_samples
pars["window"] = window
pars["model_name"] = model_name
pars["metro_adapt"] = metro_adapt
parameter_file = "training_parameters.txt"
pars["interaction"] = interaction
pars["chemical_potential"] = chemical_potential
pars["dimension"] = dimension
pars["hidden_1"] = hidden_1
n_system = n_spins ** dimension
pars["n_system"] = n_system
pars["interaction_model"] = interaction_model
print(f"The parameters for our models are\n {pars}")
if model_name == "Exact":
    data_augm = ""
    model_filename = None
    geometry_window = [window for _ in range(dimension)]
    n_system_window = window ** dimension
    ham_kin_window = get_hkin(geometry_window)
    hamiltonian_window = np.zeros((2 * n_system_window, 2 * n_system_window),
                                  dtype=np.complex128)
    sampling_fun = free_energy_exact
    sampling_args = (geometry_window, hopping, interaction, chemical_potential,
                     hamiltonian_window, ham_kin_window.data,
                     ham_kin_window.indices, ham_kin_window.indptr)
elif model_name == "N1Eigenvalues":
    n_out = 2 * window ** dimension
    in_features = 3 * window ** dimension
    if model_toggle:
        in_features += 1
    model = torch.nn.Sequential(torch.nn.Linear(in_features=in_features,
                                                out_features=hidden_1),
                                torch.nn.Tanh(),
                                torch.nn.Linear(hidden_1, n_out))
    model_filename = f"models/{data_augm}/sites_{window}_dimension_" \
                     f"{dimension}_hopping_{hopping}_interaction_" \
                     f"{interaction_model}_train_{n_train}_valid_{n_valid}.pth"
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    sampling_fun = sample_from_evals
    sampling_args = (window, dimension, interaction, chemical_potential, model)
# TODO: Add an if-loop to make a rolling window for the scanning window.
if window < n_spins:
    print("Window smaller that system size!")
    if dimension == 1:
        l_windows = np.arange(0, n_spins - window + 1, scanning_stride,
                                dtype=np.int32)
        # l_windows = np.arange(0, n_spins - window + 1, window,
        #                         dtype=np.int32)
    elif dimension == 2:
        l_windows = [(ix, iy) for ix in range(0, n_spins - window + 1,
                                                window)
                       for iy in range(0, n_spins - window + 1, window)]
else:
    if dimension == 1:
        l_windows = np.array([0], dtype=np.int32)
    elif dimension == 2:
        l_windows = np.array([(0, 0)])
n_windows = len(l_windows)
print(f"Number of windows {n_windows}!")
print(f"The window list {l_windows}.")
f_func = sample_from_window
f_args = (sampling_fun, sampling_args, window, l_windows, dimension)
print(f"Model file used {model_filename}!")
if not os.path.isdir("data"):
    os.mkdir("data")
data_dir = f"data/{model_name}"
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if model_name in "N1Eigenvalues":
    data_augm += "_"
data_filename = f"{model_name}_{data_augm}sites_{n_system}_dimension_" \
                f"{dimension}_hopping_{hopping}_interaction_{interaction}_" \
                f"samples_{n_samples}_beta_{beta}_chemicalpotential_" \
                f"{chemical_potential}_window_{window}_train_{n_train}_" \
                f"valid_{n_valid}_interactionmodel_{interaction_model}.npz"
if model_name == "Exact":
    data_filename = data_filename.split("_train")[0] + ".npz"
data_path = f"{data_dir}/{data_filename}"
accept_ratio_file = f"{data_dir}/accept_ratio_{data_filename}"
data_checkpoint = f"{data_dir}/checkpoint_{data_filename}"
if os.path.isfile(data_checkpoint):
    data_ = np.load(data_filename)
    x_ = data_["x"]
    n_samples_check = len(x_)
    n_samples -= n_samples_check
    checkpoint = np.load(data_checkpoint)
    accept_ratio = checkpoint["accept_ratio"]
    P_accept = checkpoint["P_accept"]
    P_attempt = checkpoint["P_attempt"]
    decorr_samples = int(1 / accept_ratio)
    check_point = [data_filename, decorr_samples, P_accept, P_attempt]
else:
    check_point = []
if metro_adapt:
    print("Using adaptive metropolis!")
    montecarlo_gen = metropolis_adapt(n_system, n_samples, beta, f_func,
                                      f_args, warmup=n_warmup,
                                      check_point=check_point)
else:
    print("Using metropolis without adaptation!")
    montecarlo_gen = metropolis(n_system, n_samples, beta, f_func, f_args,
                                warmup=n_warmup, check_point=check_point)
betas = beta * np.ones(n_samples)
free_energy_list = np.zeros(n_samples)
x = np.zeros((n_samples, n_system))
y = np.zeros((n_samples, n_system))
z = np.zeros((n_samples, n_system))
evals = np.zeros((n_samples, 2 * n_system))
# evecs = np.zeros((n_samples, 2 * n_system, 2 * n_system))
evecs = [] 
accept_ratio_array = np.zeros(n_samples, dtype=np.float64)
# TODO: Fix checkpointing
for i_sample in range(n_samples):
    accept_ratio, free_energy, thetas, phis, evals_vecs = next(montecarlo_gen)
    x[i_sample], y[i_sample], z[i_sample] = sph_to_cart(thetas, phis)
    free_energy /= n_system
    evals[i_sample] = evals_vecs[0]
    free_energy_list[i_sample] = free_energy
    accept_ratio_array[i_sample] = accept_ratio
    if not (i_sample % n_checkpoint):
        print(f"Sample {i_sample}")
        if i_sample:
            os.remove(data_checkpoint)
        P_attempt = int(n_warmup + i_sample)
        P_accept = int(accept_ratio * P_attempt)
        np.savez(data_checkpoint, beta=betas[:i_sample], x=x[:i_sample],
                 y=y[:i_sample], z=z[:i_sample], f=free_energy_list[:i_sample],
                 evals=evals[:i_sample], accept_ratio=accept_ratio,
                 P_attempt=P_attempt, P_accept=P_accept)
# TODO: Have a model that generates the eigenvectors as well!
np.savez(data_path, beta=betas, x=x, y=y, z=z, f=free_energy_list,
         evals=evals, evecs=evecs)
np.savez(accept_ratio_file, accept_ratio=accept_ratio_array)
os.remove(data_checkpoint)
script_duration = int(time.time()) - script_start
print(f"Script duration: {script_duration} seconds!")

