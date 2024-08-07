import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from config import results_dir_model
from config import figname, dpi, pics_dir_model


def save_fig(figname, dpi=600, pics_dir="../pics/"):
    """
    Save figure.
    """
    if not os.path.isdir(pics_dir):
        os.mkdir(pics_dir)

    plt.savefig(f"{pics_dir}/{figname}", dpi=dpi)
    plt.close()


def plot_quantity(quant_name, d_plot, figname, dpi=600, pics_dir="../pics",
                  vmin=-3, vmax=3):
    """
    Plot quantity as a function of temperature.
    """
    l_values = d_plot["l_values"]
    for d_vals in l_values:
        plt.scatter(d_vals["temperature"], d_vals["quantity_mean"],
                    c=d_vals["interaction"], vmin=vmin, vmax=vmax)
    plt.xlabel(d_plot["xlabel"])
    plt.ylabel(d_plot["ylabel"])
    plt.colorbar()
    save_fig(f"{figname}_{quant_name}", dpi, pics_dir)


def load_quantity(results_dir_model, quant_name):
    """
    Load quantities from results directory.
    """
    l_files = glob.glob(f"{results_dir_model}/*_{quant_name}_*.npz")
    l_files.sort()
    l_values = []
    for filename in l_files:
        d_pars = {}
        data = np.load(filename)
        for (key, value) in data.items():
            d_pars[key] = value
        l_values.append(d_pars)
    
    return l_values


quant_name = "avg_magnetization"
l_values = load_quantity(results_dir_model, quant_name)
d_plot = dict(l_values=l_values, xlabel="T", ylabel=r" $ <M> $")
plot_quantity(quant_name, d_plot, figname, dpi, pics_dir_model)
