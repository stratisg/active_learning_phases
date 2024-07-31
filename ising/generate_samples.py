import numpy as np
import numpy.random as rnd


def generate_ising_samples(n_samples, lattice, seed=1959):
    """
    Generate samples for Ising model.
    """
    # TODO: Generalize for arbitrary geometries.
    # Note: The current function works only for rectangular lattices 
    # and their generalizations.
    samples_shape = [n_samples]
    for sites_dim in lattice:
        samples_shape.append(sites_dim)
    samples = np.zeros(samples_shape)
    rng = rnd.default_rng(seed)
    for i_sample in range(n_samples):
        # Spin Down = 0, Spin Up = 1.
        samples[i_sample] = rng.choice([0, 1], size=lattice)
    
    return samples


def flip_spin(spin):
    """
    Flip the spin in a certain site. We assume that the spins can be 
    either 0 or 1. 
    """
    return (spin + 1) % 2


def generate_site_indices(lattice):
    """
    Generate the index for each site given a rectangular lattice.
    """
    # TODO: Generalize to arbitrary lattice geometries.
    n_sites = 1
    for sites_dim in lattice:
        n_sites *= sites_dim
    print(f"n_sites {n_sites}")
    n_dims = len(lattice) # Lattice dimensionality.
    site_indices = np.zeros((n_sites, n_dims), dtype=int)
    for dim in range(n_dims):
        for i_site in range(n_sites):
            site_indices[i_site, dim] = i_site % lattice[dim] 

    return site_indices

def calculate_ising_energy(spins, interaction=1, radius=1, external_field=0):
    """
    Calculate the enerrgy corresponding to a given spin configuration 
    for the Ising model. The only sites that interact are those within 
    a ball defined by the keyword argument radius.
    """
    lattice = spins.shape
    energy_interaction = 0
    external_field_energy = 0 

    # for dim in lattice:
    #     spin_site
    #     external_field_energy = np.dot(external_field, spin_site)


if __name__ == "__main__":
    n_samples = 1
    lattice = np.array([4, 4], dtype=int)
    samples = generate_ising_samples(n_samples, lattice)
    print(samples)
    for spin in [0, 1]:
        print(f"spin {spin}")
        spin_flip = flip_spin(spin)
        print(f"Flipped spin: {spin_flip}")
    site_indices = generate_site_indices(lattice)
    print("site_indices")
    print(site_indices)
