import os
import numpy as np
import matplotlib.pyplot as plt
from config import results_dir_model
from config import figname, dpi, pics_dir_model
from utilities import load_quantity

def save_fig(figname, dpi=600, pics_dir="../pics/"):
    """
    Save figure.
    """
    if not os.path.isdir(pics_dir):
        os.mkdir(pics_dir)

    plt.savefig(f"{pics_dir}/{figname}", dpi=dpi)
    plt.close()


def plot_quantity(quant_name, d_plot, figname, dpi=600, pics_dir="../pics",
                  vmin=0, vmax=1):
    """
    Plot quantity as a function of temperature.
    """
    l_values = d_plot["l_values"]
    for d_vals in l_values:
        plt.scatter(d_vals["interaction"], d_vals["temperature"],
                    c=d_vals["quantity_mean"], vmin=vmin, vmax=vmax)
    plt.xlabel(d_plot["xlabel"])
    plt.ylabel(d_plot["ylabel"])
    plt.colorbar()
    save_fig(f"{figname}_{quant_name}", dpi, pics_dir)


quant_name = "avg_magnetization"
l_values = load_quantity(results_dir_model, quant_name)
d_plot = dict(l_values=l_values, xlabel="J", ylabel="T")
plot_quantity(quant_name, d_plot, figname, dpi, pics_dir_model)
