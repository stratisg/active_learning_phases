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
    for label, (quantity, color, marker) in d_quantity.items():
        plt.scatter(l_temperatures, quantity, label=label, c=color,
                    marker=marker)
        save_fig(f"{figname}_{label}", dpi, pics_dir)
