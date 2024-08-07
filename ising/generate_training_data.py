"""
Generate training data to fit the model used to identify the boundary 
between phases.
"""
from simulation import Simulation
from config import data_dir
from config import model, n_samples, k_boltzmann, warmup_iter
from config import l_temperatures, l_interactions


# TODO: Convert the two for loops below to a single one that runs over
# the different configurations in order to generalize this script to
# different models.
for temperature in l_temperatures:
    print(79 * "=")
    print(f"Temperature {temperature:.3f}")
    for interaction in l_interactions:
        if not interaction:
            continue
        print(39 * "=")
        print(f"Interaction {interaction:.3f}")
        model.interaction = interaction
        simulation_args = dict(n_samples=n_samples, temperature=temperature,
                            k_boltzmann=k_boltzmann,
                            warmup_iter=warmup_iter, seed_warmup=1821,
                            seed_generation=1917, verbose=True,
                            data_dir=f"{data_dir}/{model.model_name}")
        simulation = Simulation(model, **simulation_args)
        simulation.generate_samples()
    