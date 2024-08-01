import numpy as np
import numpy.random as rnd
from analyze_samples import calculate_magnetization
from analyze_samples import calculate_absolute_magnetization


class IsingModel:
    """
    Class describing the Ising model.
    """
    def __init__(self, lattice, interaction=1, external_field=0, radius=1,
                 boundary_conds="periodic", seed=1959) -> None:
        self.model_name = "ising"
        self.lattice = lattice
        self.n_dims = len(lattice)  # Lattice dimensionality. 
        self.n_sites = 1  # Total number of sites.
        for sites_dim in lattice:
            self.n_sites *= sites_dim
        # TODO: Generalize to arbitrary lattice geometries.
        self.site_indices = np.zeros((self.n_sites, self.n_dims), dtype=int)
        self.interaction = interaction
        self.external_field = external_field
        self.radius = radius
        self.boundary_conds = boundary_conds
        self.l_neighborhood = [[] for _ in range(self.n_sites)]
        self.energy_args = {}

        # TODO: Generalize for arbitrary geometries.
        # Note: The current function works only for rectangular lattices
        # and their generalizations.
        self.spins = rnd.default_rng(seed).choice([-1, 1], size=self.lattice)

    def update(self, spin):
        """
        Flip the spin in a certain site. We assume that the spins can be 
        either -1 or 1. 
        """
        return -spin

    def generate_site_indices(self):
        """
        Generate the index for each site given a rectangular lattice.
        """
        for i_site in range(self.n_sites):
            site_ = i_site
            for dim in range(self.n_dims):
                self.site_indices[i_site, dim] = site_ % self.lattice[dim]
                site_ = site_ // self.lattice[dim]

    def get_neighborhood(self):
        """
        Identify the indices for all sites that reside inside and on the 
        boundary of the ball with the given radius centered at a given 
        site.
        """
        dist_half_lattice = np.linalg.norm(self.lattice // 2)
        for i_site, site_center in enumerate(self.site_indices):
            for site_index in self.site_indices:
                site_ = site_index.copy()
                # Euclidean distance.
                dist = np.linalg.norm(site_ - site_center)
                
                if (self.boundary_conds == "periodic" and
                dist > dist_half_lattice):
                    for dim in range(self.n_dims):
                        dist_dim = np.sqrt((site_[dim] - site_center[dim])**2)
                        half_dim = self.lattice[dim] // 2
                        if dist_dim > half_dim:
                            site_[dim] = self.lattice[dim] - site_index[dim]
        
                    # Euclidean distance.
                    dist = np.linalg.norm(site_ - site_center)
                if 0 < dist <= self.radius:
                    self.l_neighborhood[i_site].append(tuple(site_index))

    def calculate_energy(self):
        """
        Calculate the enerrgy corresponding to a given spin configuration 
        for the Ising model. The only sites that interact are those within 
        a ball defined by the keyword argument radius. We calculate the 
        average energy per site to avoid issues as we change the 
        system's size.
        """
        interaction_total = 0
        external_field_total = 0
        for i_site in range(self.n_sites):
            energy_interaction, external_field_energy = (
                self.calculate_local_energy(i_site)
                )
            interaction_total += energy_interaction
            external_field_total += external_field_energy
        
        # Correct for the double counting.
        interaction_total /= 2      
        
        return (interaction_total + external_field_total) / self.n_sites
    
    def get_spins(self):
        """
        Copy the spin at each site so we can use the copied object for 
        calculations.
        """
        spins_ = self.spins.copy()

        return spins_

    def check_align(self, spin0, spin1):
        """
        Check alignment beween two spins.
        """

        return  np.abs((spin0 + spin1) / 2)
  
    def calculate_local_energy(self, site_index):
        """
        Calculate the energy within a ball centered around the spin at
        site_index.
        """
        spin_center = self.spins[tuple(self.site_indices[site_index])]
        neighborhood = self.l_neighborhood[site_index]
        energy_interaction = 0.0
        external_field_energy = 0.0

        # Interaction energy.
        for neighbor_ in neighborhood:
            spin_ = self.spins[neighbor_]
            energy_interaction += self.check_align(spin_, spin_center)
            
        # Energy due to interaction with external field.
        external_field_energy += np.dot(self.external_field, spin_center)

        # TODO: Generalize to site-dependent interactions.
        # Multiply by interaction.
        energy_interaction *= -self.interaction

        # TODO: Do we need to take into account the double counting?
        # No! We still divide by the number of sites because the total
        # energy is the energy per site of the entire system.
        energy_local = (energy_interaction + external_field_energy)
        energy_local /=  self.n_sites

        return energy_interaction, external_field_energy
    

if __name__ == "__main__":
    lattice = np.array([4, 4], dtype=int)
    ising = IsingModel(lattice)
    print("ising.spins")
    print(ising.spins)
    ising.generate_site_indices()
    ising.get_neighborhood()
    energy = ising.calculate_energy()
    print(f"energy {energy}")
    magnetization = calculate_magnetization(ising.spins)
    print(f"magnetization {magnetization}")
    abs_magn = calculate_absolute_magnetization(ising.spins)
    print(f"absolute magnetization {abs_magn}")
