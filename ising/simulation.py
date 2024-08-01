import numpy as np
import numpy.random as rnd
from ising_model import IsingModel


# TODO: Should I convert simulation into a class?
class Simulation:
    """
    The class that desctibes each simulation.
    """
    def __init__(self, model, n_samples, temperature, k_boltzmann=1,
                 warmup_iter=int(1e3), seed_warmup=1821, seed_generation=1917):
        self.model = model
        self.n_samples = n_samples
        self.k_boltzmann = k_boltzmann
        self.temperature = temperature
        self.warmup_iter = warmup_iter
        self.seed_generation = seed_generation
        self.seed_warmup = seed_warmup
        self.decorr_length = self.model.n_sites
        self.energy = np.zeros(self.n_samples)
        self.magnetization = np.zeros_like(self.energy)
        self.magnetization_absolute = np.zeros_like(self.energy)
    
    def calculate_averages(self):
        """
        Calculate the averages of a given quantity.
        """
        return (self.energy.mean(axis=0), self.magnetization.mean(axis=0),
                self.magnetization_absolute.mean(axis=0)
                )
    
    # TODO: Save results. Name <Model>_<Model_Args>_<Quantity>
    
    def generate_samples(self):
        """
        Generate samples using Metropolis algorithm.
        """
        # Warmup stage of the Metropolis aplgorithm.
        self.metropolis_core(self.warmup_iter, self.seed_warmup,
                             warmup_stage=True)
        
        # Using Metropolis for Sample generation.
        self.metropolis_core(self.n_samples, self.seed_generation)

    def metropolis_core(self, n_iterations, seed, warmup_stage=False):
        """
        The core part of the Metropolis algorithm.
        """
        accepted_attempts = 0
        rng = rnd.default_rng(seed)
        n_total_iters = n_iterations * self.decorr_length
        for i_iter in range(n_total_iters):
            # Select a site at random to change its spin.
            site_proposed = tuple(
                self.model.site_indices[rng.choice(self.model.n_sites)]
            )
            
            # Store the current value of the spin at the proposed site.
            spin_current = self.model.spins[site_proposed].copy()

            # Store energy of the current spin configuration. 
            energy_current = self.model.calculate_energy()
            
            # Updated spin at proposed site.
            self.model.spins[site_proposed] = self.model.update(
                self.model.spins[site_proposed]
                )
            
            # Calculate the energy of the updated spin configuration.
            energy_updated = self.model.calculate_energy()
            
            # Acceptance condition
            energy_diff = energy_updated - energy_current
            # print(f"energy_diff {energy_diff}")
            # quit()
            if energy_diff <= 0:
                accepted = 1
            else:
                prob  = np.exp(-energy_diff / (self.k_boltzmann *
                                            self.temperature)
                                )
                # TODO: Consider using fixed seeds.
                prob_test = rnd.default_rng().uniform()
                accepted = prob > prob_test
            # update acceptance rate
            accepted_attempts += accepted
            
            # If the proposed spin update is not accepted, return the spin
            # configuration to its previous state.
            if not accepted:
                self.model.spins[site_proposed] = spin_current.copy()
            
            # If we are not in the warmup stage, we output the
            # generated samples.
            
            if not (i_iter % self.decorr_length) and not warmup_stage:
                i_sample = i_iter // self.decorr_length
                self.energy[i_sample] = self.model.calculate_energy()
                self.magnetization[i_sample] = (
                    self.model.calculate_magnetization()
                )
                self.magnetization_absolute[i_sample] = (
                    self.model.calculate_absolute_magnetization()
                )

        # Metropolis decorrelation length and acceptance ratio.
        decorr_length = int(n_total_iters / accepted_attempts)
        if warmup_stage:
            self.decorr_length = int(max(decorr_length, 1))
        accept_ratio = accepted_attempts / n_total_iters
        print(39 * "=")
        print(f"Acceptance ratio: {accept_ratio:3f}.")
        print(f"Number of de-correlation samples: {self.decorr_length}.")
        print(39 * "=")


def optimize():
    """
    Function for recommending parameter values to use in the subsequent
    simulation.  
    """

if __name__ == "__main__":
    ising_args = dict(lattice=np.array([4, 4], dtype=int), interaction=1,
                    external_field=0, radius=1,
                    boundary_conds="periodic", seed=1959)
    ising_model = IsingModel(**ising_args)
    ising_model.generate_site_indices()
    ising_model.get_neighborhood()

    # Simulation configuration.
    n_samples = int(1e3)
    k_boltzmann = 1
    warmup_iter = int(1e3)
    temperature = 1


    simulation = Simulation(ising_model, n_samples, temperature,
                            k_boltzmann, warmup_iter, seed_warmup=1821,
                            seed_generation=1917)
    simulation.generate_samples()
    energy_avg, magn_avg, magn_abs_avg = simulation.calculate_averages()
    print(f"energy_avg: {energy_avg}")
    print(f"magnetization avg: {magn_avg}")
    print(f"magnetization absolute avg: {magn_abs_avg}")
