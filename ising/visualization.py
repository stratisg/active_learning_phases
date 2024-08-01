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
    for label, (quantity, color, linestyle) in d_quantity.items():
        plt.plot(l_temperatures, quantity, label=label, color=color,
                 linestyle=linestyle)
    save_fig(figname, dpi, pics_dir)
