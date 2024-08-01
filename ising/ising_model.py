import numpy as np
import numpy.random as rnd


class IsingModel:
    """
    Class describing the Ising model.
    """
    def __init__(self, lattice, interaction=1, external_field=0, radius=1,
                 boundary_conds="periodic", seed=1959) -> None:
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

        # TODO: Generalize for arbitrary geometries.
        # Note: The current function works only for rectangular lattices
        # and their generalizations.
        self.spins = rnd.default_rng(seed).choice([0, 1], size=self.lattice)

    def update(self, spin):
        """
        Flip the spin in a certain site. We assume that the spins can be 
        either 0 or 1. 
        """
        return (spin + 1) % 2

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
        a ball defined by the keyword argument radius.
        """
        energy_interaction = 0.0
        external_field_energy = 0.0
        for i_site, site_center in enumerate(self.site_indices):
            energy_interaction_before = energy_interaction
            # Interaction energy.
            for neighbor_ in self.l_neighborhood[i_site]:
                spin_ = self.spins[neighbor_]
                # Check if the neighbor spin is aligned with the spin 
                # at the site_center.
                check_align = (spin_ + self.spins[tuple(site_center)] + 1) % 2
                energy_interaction += check_align
            # Energy due to interaction with external field.
            external_field_energy += np.dot(self.external_field,
                                           self.spins[tuple(site_center)])
        
        # Correct for double counting.
        energy_interaction /= 2

        # TODO: Generalize to site-dependent interactions.
        # Multiiply by interaction.
        energy_interaction *= -self. interaction

        return energy_interaction + external_field_energy


if __name__ == "__main__":
    lattice = np.array([4, 4], dtype=int)
    ising = IsingModel(lattice)
    ising.generate_site_indices()
    ising.get_neighborhood()
    energy = ising.calculate_energy()
    print(f"energy {energy}")
