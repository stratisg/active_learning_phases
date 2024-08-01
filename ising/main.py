import os
import numpy as np
from simulation import Simulation
from ising_model import IsingModel
from visualization import plot_quantity


ising_args = dict(lattice=np.array([10, 10], dtype=int), interaction=1,
                external_field=0, radius=1,
                boundary_conds="periodic", seed=1959)
ising_model = IsingModel(**ising_args)
ising_model.generate_site_indices()
ising_model.get_neighborhood()

# Simulation configuration.
n_samples = int(5e5)
k_boltzmann = 1
warmup_iter = int(1e3)

# Generate data to fit the model used to identify the boundary 
# between phases. 
temp_min = 1.5e-1
temp_max = 3.0
temp_delta = 2e-1
l_temperatures = np.arange(temp_max, temp_min, -temp_delta)
l_magn_abs_avg = np.zeros(len(l_temperatures), dtype=float)
for i_temp, temperature in enumerate(l_temperatures):
    print(79 * "=")
    print(f"Temperature {temperature:.3f}")
    simulation_args = dict(n_samples=n_samples, temperature=temperature,
                            k_boltzmann=k_boltzmann,
                            warmup_iter=warmup_iter, seed_warmup=1821,
                            seed_generation=1917)
    simulation = Simulation(ising_model, **simulation_args)
    simulation.generate_samples()
    energy_avg, magn_avg, magn_abs_avg = simulation.calculate_averages()
    l_magn_abs_avg[i_temp] = magn_abs_avg
d_quantity = {"Average absolute magnetization":
                (l_magn_abs_avg, "blue", "-")}
model_name = "ising"
figname =  f"{model_name}_average_absolute_magnetization"
dpi = 600
pics_dir = "../pics"
if not os.path.isdir(pics_dir):
    os.mkdir(pics_dir)
pics_dir_model = f"../pics/{model_name}"
# TODO: Save samples for future analysis.
# TODO: Save quantities along with temperatures.

plot_quantity(l_temperatures, d_quantity, figname, dpi=dpi,
                pics_dir=pics_dir_model)
