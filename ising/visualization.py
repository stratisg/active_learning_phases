import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from utilities import results_dir_model, model_name
from utilities import figname, dpi, pics_dir_model


def save_fig(figname, dpi=600, pics_dir="../pics/"):
    """
    Save figure.
    """
    if not os.path.isdir(pics_dir):
        os.mkdir(pics_dir)

    plt.savefig(f"{pics_dir}/{figname}", dpi=dpi)
    plt.close()


def plot_quantity(d_plot, figname, dpi=600, pics_dir="../pics"):
    """
    Plot quantity as a function of temperature.
    """
    for quant_name, (d_vals, d_plot_vars, xlabel, ylabel) in d_plot.items():
        for interaction in d_vals["interaction"]:
            plt.scatter(d_vals["l_temperatures"], d_vals["quantity"],
                        **d_plot_vars)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            save_fig(f"{figname}_{quant_name}_interaction_{interaction}", dpi,
                     pics_dir)


def load_quantity(results_dir_model, model_name):
    """
    Load quantities from results directory.
    """
    l_files = glob.glob(f"{results_dir_model}/quantities_*.npz")
    d_quantity = {}
    for filename in l_files:
        data = np.load(filename)
        quant_name = filename.split("quantity_")[1].split(".npz")[0]
        d_quantity[quant_name] = np.array(
            [
            data["l_interactions"],
            data["l_temperatures"],
            data["quantity"]
            ]
        )


d_quantities = load_quantity(results_dir_model, model_name)
d_plot = {
    "avg_magnetization": [d_quantities["avg_absolute_magn"],
                            dict(c="blue", marker="o"), "T", r" $ <M> $"
                            ],
    "avg_stagg_magnetization": [d_quantities["avg_absolute_stagg_magn"],
                                        dict(c="blue", marker="o"),
                                        "T", r" $ <M_s> $ "
                                        ]
            }
plot_quantity(d_plot, figname, dpi, pics_dir_model)