import os
import numpy as np
import matplotlib.pyplot as plt
from config import results_dir_model
from config import figname, dpi, pics_dir_model
from utilities import load_data


def save_fig(figname, dpi=600, pics_dir="../pics/"):
    """
    Save figure.
    """
    if not os.path.isdir(pics_dir):
        os.mkdir(pics_dir)

    plt.savefig(f"{pics_dir}/{figname}", dpi=dpi)
    plt.close()


def plot_quantity(quant_name, d_plot, figname, dpi=600, pics_dir="../pics",
                  vmin=0, vmax=1, marker="."):
    """
    Plot quantity as a function of temperature.
    """
    l_values = d_plot["l_values"]
    l_interaction = np.zeros(len(l_values), dtype=float)
    l_temperature = np.zeros_like(l_interaction)
    l_quantity_mean = np.zeros_like(l_interaction)
    for i_val, d_vals in enumerate(l_values):
        l_interaction[i_val] = d_vals["interaction"]
        l_temperature[i_val] = d_vals["temperature"]
        l_quantity_mean[i_val] = d_vals["quantity_mean"]
    plt.scatter(l_interaction, l_temperature, c=l_quantity_mean, vmin=vmin,
                vmax=vmax, marker=marker)
    
    plt.xlabel(d_plot["xlabel"])
    plt.ylabel(d_plot["ylabel"])
    plt.colorbar()
    save_fig(f"{figname}_{quant_name}", dpi, pics_dir)


if __name__ == "__main__":
    for quant_name in ["avg_magnetization", "avg_stagg_magnetization"]
        l_values = load_data(results_dir_model, quant_name)
        d_plot = dict(l_values=l_values, xlabel="J", ylabel="T")
        plot_quantity(quant_name, d_plot, figname, dpi, pics_dir_model)
