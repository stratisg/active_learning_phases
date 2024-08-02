import os
import matplotlib.pyplot as plt


def save_fig(figname, dpi=600, pics_dir="../pics/"):
    """
    Save figure.
    """
    if not os.path.isdir(pics_dir):
        os.mkdir(pics_dir)

    plt.savefig(f"{pics_dir}/{figname}", dpi=dpi)
    plt.close()


def plot_quantity(l_temperatures, d_quantity, figname, dpi=600,
                  pics_dir="../pics"):
    """
    Plot quantity as a function of temperature.
    """
    for quant_name, (quantity, d_plot, xlabel, ylabel) in d_quantity.items():
        plt.scatter(l_temperatures, quantity, **d_plot)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        save_fig(f"{figname}_{quant_name}", dpi, pics_dir)
