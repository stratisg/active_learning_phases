import os
import glob
import numpy as np
from ising_model import IsingModel
from ising_configuration import ising_args
from utilities import calculate_absolute_magnetization
from utilities import calculate_absolute_staggered_magnetization
from utilities import calculate_quantity_stats
from visualization import plot_quantity


# TODO: Create videos.

data_dir_parent = "../data" 
ising = IsingModel(**ising_args)
model_name = ising.model_name
ising.generate_site_indices()
data_dir_model = f"{data_dir_parent}/{model_name}"

l_filenames = glob.glob(f"{data_dir_model}/data_{model_name}*.npz")
l_temperatures = np.zeros(len(l_filenames))
avg_absolute_magn = np.zeros_like(l_temperatures)
std_absolute_magn = np.zeros_like(l_temperatures)
avg_absolute_stagg_magn = np.zeros_like(l_temperatures)
std_absolute_stagg_magn = np.zeros_like(l_temperatures)

for i_file, filename in enumerate(l_filenames):
    data = np.load(filename)
    
    # Magnetization..
    avg_absolute_magn[i_file], std_absolute_magn[i_file] = (
        calculate_quantity_stats(
        data["samples"], calculate_absolute_magnetization, {}
        )
    )

    # Staggered Magnetization..
    avg_absolute_stagg_magn[i_file], std_absolute_stagg_magn[i_file] = (
        calculate_quantity_stats(
            data["samples"],
            calculate_absolute_staggered_magnetization,
            {"site_indices": ising.site_indices}
        )
    )

    l_temperatures[i_file] = data["temperature"]

d_quantity = {
    "average_magnetization": [avg_absolute_magn,
                              dict(c="blue", marker="o"), "T", r" $ <M> $"
                              ],
    "std_magnetization": [std_absolute_magn,
                          dict(c="red", marker="P"), "T", r" $ C $ "
                          ],
    "average_staggered_magnetization": [avg_absolute_stagg_magn,
                                        dict(c="blue", marker="o"),
                                        "T", r" $ <M_s> $ "
                                        ],
    "std_staggered_magnetization": [std_absolute_stagg_magn,
                                    dict(c="red", marker="P"),
                                     "T", r" $ <C_s> $ "
                                    ]
            }

figname =  f"{model_name}"
dpi = 600
pics_dir = "../pics"
if not os.path.isdir(pics_dir):
    os.mkdir(pics_dir)
pics_dir_model = f"../pics/{model_name}"
plot_quantity(l_temperatures, d_quantity, figname, dpi, pics_dir_model)
