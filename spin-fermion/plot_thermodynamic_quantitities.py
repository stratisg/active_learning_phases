import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import argparse
import glob
import os
from utilities import generate_filename
from training_parameters import n_train, n_valid
from training_parameters import data_augm
from generate_samples_config import dimension, hopping, interaction
from generate_samples_config import chemical_potential
from generate_samples_config import n_samples, model_name, data_augm
from generate_samples_config import interaction_model


def load_thermodynamic_quantities(data_files):
    """Load data from quantity files."""
    data_files.sort()
    quantities = np.zeros((n_data, len(fig_dict) - 2))
    error = np.zeros_like(quantities)
    betas = np.zeros(len(data_files))
    spin_corr = []
    error_spin_corr = []
    spin_struct = []
    error_spin_struct = []
    for i_file, data_file in enumerate(data_files):
        data_ = np.load(data_file)
        betas[i_file] = float(data_file.split("beta_")[1].split("_")[0])
        quantities[i_file] = data_["quantities"]
        error[i_file] = data_["errors"]
        spin_corr_file = data_file.replace("quantities", "spin_correlation")
        data_corr = np.load(spin_corr_file)
        spin_corr.append(data_corr["spin_corr"])
        error_spin_corr.append(data_corr["spin_corr_errors"])
        spin_struct_file = data_file.replace("quantities",
                                             "spin_structure_factor")
        data_struct = np.load(spin_struct_file)
        spin_struct.append(data_struct["spin_struct"])
        error_spin_struct.append(data_struct["spin_struct_errors"])
    r_vector = data_corr["r_vector"]
    momenta = data_struct["momenta"]
    d_quantities = dict(quantities=quantities, spin_corr=spin_corr,
                        r_vector=r_vector, spin_struct=spin_struct,
                        momenta=momenta)
    d_error = dict(error=error, error_spin_corr=error_spin_corr,
                   error_spin_struct=error_spin_struct)

    return d_quantities, d_error, betas


script_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("n_spins", type=int,
                    help="Number of lattice sites per dimension.")
parser.add_argument("window", type=int,
                    help="Window used for the sample generation.")
parser.add_argument("importance_sampling", type=int, choices=[0, 1], default=1,
                    help="0: w/o importance sampling,"
                         "1: w/ importance sampling")
parser_args = parser.parse_args()
n_spins = parser_args.n_spins
window = parser_args.window
importance_sampling = parser_args.importance_sampling
data_dir = "docs/generate_curves/"
fig_dict = [r"$\langle E \rangle$", r"$C_v$", r"$|\mathbf{M}|$",
            r"$|\mathbf{M}_s|$", r"$C(r)$", r"$S(\mathbf{q})$"]
matplotlib.rcParams["markers.fillstyle"] = "none"
matplotlib.rcParams["lines.linewidth"] = "0.5"
matplotlib.rcParams["lines.markersize"] = "4.5"
matplotlib.rcParams["text.usetex"] = "True"
matplotlib.rcParams["axes.labelsize"] = "10"
matplotlib.rcParams["xtick.labelsize"] = "8"
matplotlib.rcParams["xtick.top"] = "True"
matplotlib.rcParams["ytick.right"] = "True"
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
l_model = ["Exact", "N1Eigenvalues"]
l_label = ["Exact", "N1Eigenvalues"]
l_marker = ["o", "*"]
l_color = ["blue", "red"]
n_quant = 4
n_figs = n_quant - 2 * (1 - importance_sampling)
n_corr = 4
n_struct = 3
line_width = 3.375
aspect_ratio = 3.5
figsize_quant = (line_width, n_figs * line_width / aspect_ratio)
figsize_corr = (line_width, n_corr * line_width / aspect_ratio)
figsize_struct = (line_width, n_struct * line_width / aspect_ratio)
fig_quant, ax_quant = plt.subplots(n_figs, 1, figsize=figsize_quant,
                                   sharex=True)
fig_quant_diff, ax_quant_diff = plt.subplots(n_figs, 1, figsize=figsize_quant,
                                             sharex=True)
fig_corr, ax_corr = plt.subplots(n_corr, 1, figsize=figsize_corr, sharex=True)
for i_ax in range(n_figs):
    ax_quant[i_ax].set_xscale("log")
    ax_quant[i_ax].tick_params(axis="both")
    ax_quant_diff[i_ax].set_xscale("log")
    ax_quant_diff[i_ax].set_yscale("log")
    ax_quant_diff[i_ax].tick_params(axis="both")
for i_ax in range(n_corr):
    ax_corr[i_ax].set_xscale("log")
    ax_corr[i_ax].tick_params(axis="both")
ax_corr[-1].set_xlabel("T")
l_temps = [1.0, 0.5, 0.1]
beta = "*"
for i_model, model_name in enumerate(l_model):
    print(f"Model {model_name}.")
    if model_name == "Exact":
        data_augm_ = ""
    else:
        data_augm_ = data_augm
    filehead = generate_filename(model_name, data_augm_, n_spins, dimension,
                                 hopping, interaction, n_samples, beta,
                                 chemical_potential, window, n_train, n_valid,
                                 interaction_model)
    quantities_head  = "quantities"
    if not importance_sampling:
        quantities_head += "_noimportance"
    file_template = f"docs/generate_curves/{quantities_head}_{filehead}"
    data_files = glob.glob(file_template)
    n_data = len(data_files)
    print(f"Number of data files {n_data}.")
    if not n_data:
        continue
    d_quantities, d_error, betas = load_thermodynamic_quantities(data_files)
    label = l_label[i_model]
    marker = l_marker[i_model]
    color = l_color[i_model]
    print(f"betas {betas}")
    temps = 1 / betas
    quantities = d_quantities["quantities"].T
    if model_name == "Exact":
        temps_exact = temps
        quant_exact = quantities
    error_quant = d_error["error"].T
    spin_corr = np.array(d_quantities["spin_corr"])
    error_spin_corr = d_error["error_spin_corr"]
    r_vector = np.array(d_quantities["r_vector"])
    spin_struct = np.array(d_quantities["spin_struct"])
    error_spin_struct = d_error["error_spin_struct"]
    momenta = d_quantities["momenta"]
    symmetry_pts = np.arange(len(momenta))
    # TODO: Put errors.
    for i_r_vec, r_vec in enumerate(r_vector[:n_corr]):
        ax_corr[i_r_vec].scatter(temps, spin_corr[:, i_r_vec], marker=marker,
                                 label=label, color=color)
    for i_ax, quantity in enumerate(fig_dict[n_quant - n_figs: n_quant]):
        i_quantity = i_ax + n_quant - n_figs
        ax_quant[i_ax].set_ylabel(quantity)
        ax_quant[i_ax].yaxis.set_label_position("right")
        ax_quant[i_ax].scatter(temps, quantities[i_quantity], marker=marker,
                               label=label, color=color)
        if model_name != "Exact":
            quant_diff = quantities[i_quantity] - quant_exact[i_quantity]
            quant_diff = np.abs(quant_diff / quant_exact[i_quantity])
            ax_quant_diff[i_ax].set_ylabel(quantity)
            ax_quant_diff[i_ax].yaxis.set_label_position("right")
            ax_quant_diff[i_ax].scatter(temps, quant_diff, marker=marker,
                                        label=label, color=color)
for i_r_vec, r_vec in enumerate(r_vector[:n_corr]):
    ax_corr[i_r_vec].set_ylabel(r"$C(\mathbf{r}$ = " + f"{r_vec})")
    ax_corr[i_r_vec].yaxis.set_label_position("right")
ax_quant[-1].set_xlabel("T")
ax_quant_diff[-1].set_xlabel("T")
fig_quant.align_ylabels(ax_quant)
fig_corr.align_ylabels(ax_corr)
ax_quant[-1].legend(frameon=False, fontsize=6)
ax_corr[-1].legend(frameon=False, fontsize=6)
hspace = 0.05
top = 0.98
bottom = 0.2
if importance_sampling:
    bottom *= 0.5
fig_quant.subplots_adjust(hspace=hspace, top=top, bottom=bottom)
fig_quant_diff.subplots_adjust(hspace=hspace, top=top, bottom=bottom)
fig_corr.subplots_adjust(hspace=hspace, top=top, bottom=bottom)
if not os.path.isdir("pics"):
    os.mkdir("pics")
dir_name = "pics/analysis"
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
filehead = filehead.replace("beta_*", "")
filehead = filehead.replace(".npz", ".png")
for fig in ["quantities", "quant_diff", "correlation", "structure"]:
    figname = fig
    if not importance_sampling:
        figname += "_noimportance"
    fig_path = f"{dir_name}/{figname}_{filehead}"
    if fig == "quantities":
        fig_quant.savefig(fig_path, dpi=600)
    elif fig == "quant_diff":
        fig_quant_diff.savefig(fig_path, dpi=600)
    elif fig == "correlation":
        fig_corr.savefig(fig_path, dpi=600)
script_duration = int(time.time() - script_start)
print(f"Script duration: {script_duration} seconds!")

