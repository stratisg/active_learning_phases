import numpy as np


def get_boltzmann_dist(free_energy, beta):
    """Calculate the partition function based on the Monte-Carlo
        sichemical_potentiallations."""
    weight = np.exp(-beta * free_energy)
    partition_function = weight.sum()

    return weight, partition_function


def get_free_energy(beta, evals, chemical_potential):
    """Get the free energy given the inverse temperature 'beta', the
       eigenvalues of the Hamiltonian 'evals',
       and the chemical potential 'chemical_potential'."""
    free_energy_list = np.logaddexp(0, -beta * (evals - chemical_potential),
                                    dtype=np.float128)
    if free_energy_list.ndim > 1:
        free_energy = - free_energy_list.sum(axis=1) / beta
    else:
        free_energy = - free_energy_list.sum() / beta

    return free_energy


def get_importance_sampling(l_free_energy, eigen_energies, n_spins, dimension,
                            beta, chemical_potential):
    """Generate weights according to importance sampling."""
    n_system = n_spins ** dimension
    l_free_energy *= n_system
    l_free_energy_exact = get_free_energy(beta, eigen_energies, chemical_potential)
    l_delta_free = l_free_energy_exact - l_free_energy
    # TODO: Make this function agnostic to the underlying probability 
    #   distribution.
    weight, partition_function = get_boltzmann_dist(l_delta_free, beta)
    weight /= partition_function

    return weight, partition_function

