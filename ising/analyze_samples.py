import os
import glob
import numpy as np
from visualization import plot_quantity


def calculate_magnetization(spin_config):
    """
    Calculate the system's magnetization per site.
    """ 

    return spin_config.mean()

def calculate_absolute_magnetization(spin_config):
    """
    Calculate the absolute value of the system's magnetization per site.
    """

    return np.abs(calculate_magnetization(spin_config))

def calculate_quantity_stats(l_samples, quantity_fn, quant_args):
    """
    Calculate the sample mean and sample standard deviation for the 
    given quantity. 
    """
    quantity = np.zeros(len(l_samples))
    # TODO: Consider using map().
    for i_sample, sample in enumerate(l_samples):
        quantity[i_sample] = quantity_fn(sample, **quant_args)

    return quantity.mean(), quantity.std()

# TODO: Create videos.

data_dir_parent = "data" 
model_name = "ising"
data_dir_model = f"{data_dir_parent}/{model_name}"

l_filenames = glob.glob(f"{data_dir_model}/data_{model_name}*.npz")
l_temperatures = np.zeros(len(l_filenames))
avg_absolute_magn = np.zeros_like(l_temperatures)
std_absolute_magn = np.zeros_like(l_temperatures)
for i_file, filename in enumerate(l_filenames):
    data = np.load(filename)
    avg_absolute_magn[i_file], std_absolute_magn[i_file] = (
        calculate_quantity_stats(
        data["samples"], calculate_absolute_magnetization, {}
        )
    )
    l_temperatures[i_file] = data["temperature"]
d_quantity = {"Average_M": [avg_absolute_magn, "blue", "o"],
              "Std_M": [std_absolute_magn, "red", "P"]}

figname =  f"{model_name}_absolute_magnetization"
dpi = 600
pics_dir = "../pics"
if not os.path.isdir(pics_dir):
    os.mkdir(pics_dir)
pics_dir_model = f"../pics/{model_name}"
plot_quantity(l_temperatures, d_quantity, figname, dpi, pics_dir_model)
