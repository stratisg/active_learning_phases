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
        for i_site, site_center in enumerate(self.site_indices):
            for site_index in self.site_indices:
                site_ = site_index.copy()
                if self.boundary_conds == "periodic":
                    for dim in range(self.n_dims):
                        print(f"dim {dim}")
                        print(f"site_[dim] {site_[dim]}")
                        print(f" self.lattice[dim] {self.lattice[dim]}")
                        
                        if site_[dim] >= (self.lattice[dim] // 2):
                            site_[dim] = site_index[dim] - self.lattice[dim] // 2
                # Euclidean distance.
                if 0 < np.linalg.norm(site_ - site_center) <= self.radius:
                    self.l_neighborhood[i_site].append(site_index)

    def calculate_energy(self):
        """
        Calculate the enerrgy corresponding to a given spin configuration 
        for the Ising model. The only sites that interact are those within 
        a ball defined by the keyword argument radius.
        """
        energy_interaction = 0
        external_field_energy = 0 
        for i_site, site_center in enumerate(self.site_indices):
            # Identify sites within the ball of given radius.
            pass
            # self.l_neighborhood[i_site].sum() * self.spins[site]
            #     external_field_energy = np.dot(external_field, spin_site)


if __name__ == "__main__":
    lattice = np.array([4, 4], dtype=int)
    ising = IsingModel(lattice)
    
    for spin in [0, 1]:
        print(f"spin {spin}")
        spin_flip = ising.update(spin)
        print(f"Flipped spin: {spin_flip}")
    ising.generate_site_indices()
    print("site_indices")
    print(ising.site_indices)
    ising.get_neighborhood()
    print("l_neighborhood")
    print(ising.l_neighborhood)
