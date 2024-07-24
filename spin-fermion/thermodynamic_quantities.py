import numpy as np


def get_variance(l_values, mean_value, weights):
    """Calculate the variance of 'l_values' given the corresponding
        weights for each of its elements."""
    squared_error = (l_values - mean_value) ** 2
    weighted_squared_error = weights * squared_error
    weighted_variance = weighted_squared_error.sum()

    return weighted_variance


def get_fermi_distribution(eigen_energies, beta, chemical_potential):
    """Fermi-Dirac distribution."""
    # The Fermi distribution formula in the line below:
    # fermi_distribution = 1 / (1 + np.exp(beta * (eigen_energies - \
    #                                              chemical_potential)))
    # The expression below is used to avoid overflow issues at high beta
    # values. This is Phil Weinberg's suggestion.
    energies_mod = eigen_energies - chemical_potential
    fermi_distribution = np.exp(-np.logaddexp(0, beta * energies_mod))

    return fermi_distribution


def find_coords(site_index, dimension, length):
    """Given the site index of a flatten N-dimensional lattice site 
       we find its cartesian coordiantes in the original lattice.
       site_index is a scalar.
       length is the length of the lattice on one of its dimensions 
       assuming a cubical lattice."""
    # TODO: Give length as a vector so we can work with rectangular 
    #       lattices.
    powers = np.array([length ** ind
                       for ind in range(dimension - 1, -1, -1)])
    # TODO: Make it vector compatible
    coords = np.zeros(dimension, dtype=np.int32)
    for i_dim in range(dimension):
        coords[i_dim] = site_index // powers[i_dim]
        site_index %= powers[i_dim]

    return coords


def find_site_index(coords, length):
    """Given the Cartesian cordinates of a lattice site in N-dimensions
       we find the index of the site if the lattice was reshaped as an 
       1D array.
       coords is an N-tuple or a column vector with N entries
       length is the length of the lattice on one of its dimensions 
       assuming a cubical lattice."""
    # TODO: Raise error if any of the coords is bigger than the length
    #      across that dimension.
    coords = np.array(coords)
    # The components are the column values.
    dimension = len(coords)
    # TODO: give length as a vector so we can work with rectangular lattices.
    powers = np.array([length ** ind for ind in range(dimension - 1, -1, -1)])
    site_index_ = coords * powers
    site_index = site_index_.sum()

    return site_index


def find_indices_for_linear(dimension, n_spins, print_toggle=False):
    """ Find indices for linear model needed for the dot product."""
    site_list = np.arange((n_spins // 2 + 1) ** dimension)
    dist_list = np.zeros(len(site_list))
    coords_list = np.zeros(((n_spins // 2 + 1) ** dimension, dimension),
                           dtype=np.int32)
    for i_site in site_list:
        coords = find_coords(i_site, dimension, n_spins // 2 + 1)
        coords_list[i_site] = coords
        dist_list[i_site] = np.dot(coords, coords) ** 0.5
    r_list = np.sort(dist_list)
    r_l = []
    i_list = []
    for radius in r_list:
        if not radius in r_l and radius:
            r_l.append(radius)
            ind_ = np.asarray(dist_list == radius).nonzero()
            i_list.append(coords_list[ind_])
    spin_list = [tuple(find_coords(i_site, dimension, n_spins))
                 for i_site in range(n_spins ** dimension)]
    if print_toggle:
        print(f"The distances between neighbors are {r_l}!")
        print(f"The indices to the corresponding distances are {i_list}!")
        print(f"spin_list {spin_list}")

    return i_list, spin_list, r_l


def get_average_energy(eigen_energies, weights, beta, chemical_potential):
    """Average energy for spin-fermion model."""
    # TODO: In the future normalize the energy with the system's size.
    fermi_dist = get_fermi_distribution(eigen_energies, beta,
                                        chemical_potential)
    l_avg_energy_fermi = (eigen_energies - chemical_potential) * fermi_dist
    avg_energy_fermi = l_avg_energy_fermi.sum(axis=1)
    weighted_fermi_energy = weights * avg_energy_fermi
    avg_energy = weighted_fermi_energy.sum()
    # TODO: Check the formula for avg_energy_error.
    avg_energy_var = get_variance(avg_energy_fermi, avg_energy, weights)
    avg_energy_error = np.sqrt(avg_energy_var)
    # TODO: Do we need an error for the average Fermi energy?

    return avg_energy, avg_energy_error, avg_energy_fermi


def get_specific_heat(eigen_energies, weights, beta, chemical_potential):
    """Specific heat function for spin-fermion model."""
    # TODO: In the future normalize the specific heat with the
    #       system's size.
    fermi_dist = get_fermi_distribution(eigen_energies, beta,
                                        chemical_potential)
    diff_energies = (eigen_energies - chemical_potential)
    l_avg_energy = get_average_energy(eigen_energies, weights, beta,
                                      chemical_potential)
    avg_energy = l_avg_energy[0]
    avg_energy_fermi = l_avg_energy[2]
    fermi_correction = diff_energies ** 2 * fermi_dist * (1 - fermi_dist)
    fermi_correction = fermi_correction.sum(axis=1)
    fermi_part = avg_energy_fermi ** 2 + fermi_correction
    weighted_left_part = weights * fermi_part
    left_part = weighted_left_part.sum()
    specific_heat = beta ** 2 * (left_part - avg_energy ** 2)
    l_spec_heat = beta ** 2 * (fermi_part - avg_energy ** 2)
    specific_heat_var = get_variance(l_spec_heat, specific_heat, weights)
    specific_heat_error = np.sqrt(specific_heat_var)

    return specific_heat, specific_heat_error


def get_abs_magnetization(l_spin_components, weights):
    """Gives the magnitude of the average magnetization vector."""
    # TODO: Changed sum to mean in the following three lines.
    magn_x = l_spin_components[0].mean(axis=1)
    magn_y = l_spin_components[1].mean(axis=1)
    magn_z = l_spin_components[2].mean(axis=1)
    unweighted_abs_magn = np.sqrt(magn_x**2 + magn_y**2 + magn_z**2)
    weighted_abs_magn = weights * unweighted_abs_magn
    abs_magn = weighted_abs_magn.sum()
    # TODO: Check the error formula below.
    abs_magn_var = get_variance(unweighted_abs_magn, abs_magn, weights)
    abs_magn_error = np.sqrt(abs_magn_var)

    return abs_magn, abs_magn_error


def get_stagg_magnetization(l_spin_components, n_spins, dimension, weights):
    """Calculate the staggered magnetization."""
    n_samples = len(weights)
    stag_shape = n_spins * np.ones(dimension, dtype=np.int32)
    stag_flat = np.arange(l_spin_components.shape[2])
    stag = np.zeros(stag_shape)
    for site_index in stag_flat:
        coords = find_coords(site_index, dimension, n_spins)
        stag[tuple(coords)] = (-1) ** (coords.sum())
    stag = stag.reshape(-1)
    unweighted_stagg_magn = np.zeros(n_samples)
    for i_sample in range(n_samples):
        spin_components = l_spin_components[:, i_sample]
        magn_x = np.mean(spin_components[0] * stag)
        magn_y = np.mean(spin_components[1] * stag)
        magn_z = np.mean(spin_components[2] * stag)
        unweighted_stagg_magn[i_sample] = np.sqrt(magn_x ** 2 + magn_y ** 2 +
                                                  magn_z ** 2)
    weighted_stagg_magn = weights * unweighted_stagg_magn
    stagg_magn = weighted_stagg_magn.sum()
    stagg_magn_var = get_variance(unweighted_stagg_magn, stagg_magn, weights)
    stagg_magn_error = np.sqrt(stagg_magn_var)

    return stagg_magn, stagg_magn_error


def get_spin_correlation(l_spin_components, n_spins, dimension, weights):
    """Get the average spin correlation for a certain ensemble."""
    n_samples = l_spin_components.shape[1]
    i_list, spin_list, r_l = find_indices_for_linear(dimension, n_spins)
    l_r_vector = []
    for i_distance in range(len(i_list)):
        for element in i_list[i_distance]:
            l_r_vector.append(element)
    unweighted_spin_correlation = np.zeros((n_samples, len(l_r_vector)))
    for i_vector, r_vector in enumerate(l_r_vector):
        print(f"r_vector {r_vector}")
        for i_sample in range(n_samples):
            spin_comps_ = l_spin_components[:, i_sample].T
            dot_product = np.zeros(len(spin_list))
            for i_site, site in enumerate(spin_list):
                i0 = find_site_index(site, n_spins)
                site_1 = tuple((site + r_vector) % n_spins)
                i1 = find_site_index(site_1, n_spins)
                dot_product[i_site] = np.dot(spin_comps_[i0], spin_comps_[i1])
            sample_spin_corr = dot_product.mean()
            if not np.sum(r_vector % (n_spins // 2)):
                sample_spin_corr /= 2
            unweighted_spin_correlation[i_sample, i_vector] = sample_spin_corr
    weighted_spin_correlation = np.zeros((n_samples, len(l_r_vector)))
    for i_r in range(len(l_r_vector)):
        weighted_spin_correlation[:, i_r] = weights * \
                                            unweighted_spin_correlation[:, i_r]
    spin_correlation = weighted_spin_correlation.sum(axis=0)
    spin_correlation_var = np.zeros_like(spin_correlation)
    for i_r in range(len(l_r_vector)):
        unw_spin_corr_ = unweighted_spin_correlation[:, i_r]
        spin_correlation_ = spin_correlation[i_r]
        spin_correlation_var[i_r] = get_variance(unw_spin_corr_,
                                                 spin_correlation_, weights)
    spin_correlation_error = np.sqrt(spin_correlation_var)

    return spin_correlation, spin_correlation_error, l_r_vector


def get_spin_structure(momentum, l_spin_components,  n_spins, dimension,
                       weights, l_spin_corr=None, boundary_cond="periodic"):
    """Gives us the spin structure factor for a given momentum vector."""
    if l_spin_corr is None:
        l_spin_corr = get_spin_correlation(l_spin_components, n_spins,
                                           dimension, weights)
    spin_corr = l_spin_corr[0]
    spin_corr_err = l_spin_corr[1]
    l_r_vec = l_spin_corr[2]
    l_structure_factor = 2 * np.exp(-1j * np.dot(momentum, l_r_vec.T)) * \
                         spin_corr
    structure_factor = l_structure_factor.sum()
    # TODO: Check the error formula below. Remove placeholder below.
    structure_factor_error = 0
    # structure_factor_var = get_variance(l_structure_factor, structure_factor,
    #                                     weights)
    # structure_factor_error = np.sqrt(structure_factor_var)

    return structure_factor, structure_factor_error
    
