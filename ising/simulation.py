import os
import numpy as np
import numpy.random as rnd


class Simulation:
    """
    The class that desctibes each simulation.
    """
    def __init__(self, model, n_samples, temperature, k_boltzmann=1,
                 warmup_iter=int(1e3), seed_warmup=1821, seed_generation=1917,
                 verbose=False, data_dir="data"):
        
        self.model = model

        self.n_samples = n_samples
        self.k_boltzmann = k_boltzmann
        self.temperature = temperature
        self.warmup_iter = warmup_iter
        self.seed_generation = seed_generation
        self.seed_warmup = seed_warmup
        self.verbose = verbose

        self.decorr_length = self.model.n_sites
        samples_shape = np.append([n_samples], self.model.lattice)
        self.samples = np.zeros(samples_shape)
        self.data_dir = data_dir
    
    def generate_samples(self):
        """
        Generate samples using Metropolis algorithm.
        """
        # Warmup stage of the Metropolis aplgorithm.
        self.metropolis_core(self.warmup_iter, self.seed_warmup,
                             warmup_stage=True, verbose=self.verbose)
        
        # Using Metropolis for Sample generation.
        self.metropolis_core(self.n_samples, self.seed_generation,
                             verbose=self.verbose)
        
        # Save data.
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        
        np.savez(f"{self.data_dir}/data_{self.model.model_name}_"\
                 f"temperature_{self.temperature}_"\
                 f"interaction_{self.model.interaction}.npz",
                 samples=self.samples,
                 temperature=self.temperature,
                 interaction=self.model.interaction,
                 external_field=self.model.external_field,
                 neighborhood_radius=self.model.radius)
        # TODO: Is there a smart way to save all the attributes of a 
        # class without writing them out explicitly?

    def metropolis_core(self, n_iterations, seed, warmup_stage=False,
                        verbose=False):
        """
        The core part of the Metropolis algorithm.
        """
        accepted_attempts = 0
        rng = rnd.default_rng(seed)
        n_total_iters = n_iterations * self.decorr_length
        for i_iter in range(n_total_iters):
            # Select a site at random to change its spin.
            site_proposed_index = rng.choice(self.model.n_sites)
            site_proposed = tuple(self.model.site_indices[site_proposed_index])
            
            # Store the current value of the spin at the proposed site.
            spin_current = self.model.spins[site_proposed].copy()

            # Store energy of the current spin configuration.
            # TODO: Write a method energy difference in each model.
            # TODO: Make this calculation model agnostic. Right now it
            # is written specifically for the Ising model.
            interaction_curr, external_curr = (
                self.model.calculate_local_energy(site_proposed_index)
                )
            local_energy_current = interaction_curr + external_curr

            # Updated spin at proposed site.
            self.model.spins[site_proposed] = self.model.update(
                self.model.spins[site_proposed]
                )
            
            # Calculate the energy of the updated spin configuration.
            interaction_update, external_update = (
                self.model.calculate_local_energy(site_proposed_index)
            )
            local_energy_updated = interaction_update + external_update

            # Acceptance condition
            # energy_diff = energy_updated - energy_current
            energy_diff = local_energy_updated - local_energy_current

            if energy_diff <= 0:
                accepted = 1
            else:
                prob  = np.exp(-energy_diff / (self.k_boltzmann *
                                               self.temperature)
                            )
                # TODO: Consider using fixed seeds.
                prob_test = rnd.default_rng().uniform()
                accepted = prob > prob_test
            
            # Update acceptance rate
            accepted_attempts += accepted
            
            # If the proposed spin update is not accepted, return the spin
            # configuration to its previous state.
            if not accepted:
                self.model.spins[site_proposed] = spin_current.copy()
            
            # If we are not in the warmup stage, we output the
            # generated samples.
            if not (i_iter % self.decorr_length) and not warmup_stage:
                i_sample = i_iter // self.decorr_length
                self.samples[i_sample] = self.model.spins

        # Metropolis decorrelation length and acceptance ratio.
        decorr_length = int(n_total_iters / accepted_attempts)
        if warmup_stage:
            self.decorr_length = int(max(decorr_length, 1))
        accept_ratio = accepted_attempts / n_total_iters
        if verbose:
            print(39 * "=")
            print(f"Acceptance ratio: {accept_ratio:3f}.")
            print(f"Number of de-correlation samples: {self.decorr_length}.")
            print(39 * "=")


if __name__ == "__main__":
    from config import model, data_dir_model


    # Simulation configuration.
    n_samples = int(1e3)
    k_boltzmann = 1
    warmup_iter = int(1e3)
    temperature = 2.0
    simulation_args = dict(n_samples=n_samples, temperature=temperature,
                           k_boltzmann=k_boltzmann, warmup_iter=warmup_iter,
                           seed_warmup=1821, seed_generation=1917,
                           verbose=True, data_dir=data_dir_model)
    simulation = Simulation(model, **simulation_args)
    simulation.generate_samples()
