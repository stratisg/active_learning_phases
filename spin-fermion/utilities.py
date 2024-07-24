import os
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score


def generate_filename(modelname, data_augm, sites, dimension, hopping,
                     interaction, samples, beta, chemical_potential, window,
                     train, valid, interaction_model):
    """Generate the filename template that is used across the different 
        scripts."""
    if modelname == "N1Eigenvalues":
        data_augm += "_"
    filename = f"{modelname}_{data_augm}sites_{sites}_dimension_" \
               f"{dimension}_hopping_{hopping}_interaction_{interaction}_" \
               f"samples_{samples}_beta_{beta}_chemicalpotential_" \
               f"{chemical_potential}_window_{window}_train_{train}_valid_" \
               f"{valid}_interactionmodel_{interaction_model}.npz"

    return filename


def appropriate_shape(filename, dimension):
    """
    Outputs data with the correct shape based on the system's geometry.
    """
    data = np.load(filename)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    n_samples = int(filename.split("samples_")[1].split("_")[0])
    n_spins = round(np.power(x.shape[1], 1 / dimension))
    if dimension == 1:
        coords = np.zeros((n_samples, 3, n_spins))
        for i_sample in range(n_samples):
            coords[i_sample, 0] = x[i_sample]
            coords[i_sample, 1] = y[i_sample]
            coords[i_sample, 2] = z[i_sample]
    elif dimension > 1:
        lattice_shape = n_spins * np.ones(dimension, dtype=np.int32)
        coord_shape = np.array([n_samples, 3], dtype=np.int32)
        coord_shape = np.append(coord_shape, lattice_shape)
        coords = np.zeros(coord_shape)
        for i_sample in range(n_samples):
            coords[i_sample, 0] = x[i_sample].reshape(lattice_shape)
            coords[i_sample, 1] = y[i_sample].reshape(lattice_shape)
            coords[i_sample, 2] = z[i_sample].reshape(lattice_shape)
    evals = data["evals"]
    evecs = data["evecs"]

    return coords, evals, evecs


def batch_data(in_data, out_data, size=100):
    """
    Choose a random batch of size "size" from the training data set. An epoch
    is when we use all the data batches from our training set.
    """
    random_order = np.random.choice(len(in_data), size=size, replace=False)
    in_data = in_data[random_order]
    out_data = out_data[random_order]

    return in_data, out_data


def cart_to_sph(x, y, z, r=1) -> object:
    """ Convert cartesian to spherical coordinates."""
    # TODO: Can I not use r = np.sqrt(x**2 + y**2 + z**2)? Probably issue
    #       with x,y, and z being matrices and they all have the same r.
    theta = np.arccos(z / r)
    phi = np.arccos(x / (np.sqrt(r ** 2 - z ** 2) + 1e-8))
    ind = np.where(y < 0)
    if len(ind[0]):
        phi[ind] = 2 * np.pi - phi[ind]

    return theta, phi


def sph_to_cart(theta, phi, r=1):
    """ Convert spherical coordinates to cartesian."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def lattice_geometry(n_spins, dimension):
    """Outputs the lattice geometry."""
    geometry = n_spins * np.ones(dimension, dtype=int)
    
    return geometry


def flatten_data(data, n_spins, dimension):
    """"Turns the data matrix into an 1D array where the x, y, and z
    components of the ith site are located at 3 * i, 3 * i + 1, and 3 * i + 2
    respectively."""
    # TODO: Is this function independent of the system's dimensionality?
    n_system = n_spins ** dimension
    flat_data = np.zeros((len(data), 3 * n_system))
    for i_datum in range(len(data)):
        x = data[i_datum, 0].reshape(-1)
        y = data[i_datum, 1].reshape(-1)
        z = data[i_datum, 2].reshape(-1)
        for i_site in range(n_system):
            flat_data[i_datum, 3 * i_site] = x[i_site]
            flat_data[i_datum, 3 * i_site + 1] = y[i_site]
            flat_data[i_datum, 3 * i_site + 2] = z[i_site]

    return flat_data


def apply_discrete_translation1d(input_data, evals, dimension,
                                 n_spins, n_samples):
    """Apply data augmentation based on the discrete translation symmetry of
    our Hamiltonian."""
    print("We are using the discrete translation invariance of the "
          "Hamiltonian to augment our data!")
    n_system = n_spins ** dimension
    data_augm_shape = [input_data.shape[i] for i in range(len(input_data.shape))]
    data_augm_shape[0] *= n_system
    in_augm_train = np.zeros(data_augm_shape)
    evals_augm_train = np.zeros([n_samples * n_system, 2 * n_system])
    per_mat = np.zeros((n_spins, n_spins))
    z = np.eye(n_spins)
    for i_site in range(n_spins):
        per_mat[:, i_site] = z[:, i_site - 1]
    for i_sample in range(n_samples):
        data_set = input_data[i_sample]
        for i_x in range(n_spins):
            i_augm = i_sample * n_system + i_x
            in_augm_train[i_augm] = data_set
            evals_augm_train[i_augm] = evals[i_sample]
            data_set = np.matmul(data_set, per_mat)

    return in_augm_train, evals_augm_train


# TODO: We need a discrete tranlastion function independent of the 
#       system's dimensionality.
def apply_discrete_translation2d(input_data, evals, dimension, n_spins,
                                 n_samples):
    """Apply data augmentation based on the discrete translation symmetry of
    our Hamiltonian."""
    print("We are using the discrete translation invariance of the "
          "Hamiltonian to augment our data!")
    data_shape = [n_samples, 3, n_spins, n_spins]
    input_data = input_data.reshape(data_shape)
    n_system = n_spins ** dimension
    data_augm_shape = data_shape.copy()
    data_augm_shape[0] *= n_system
    in_augm_train = np.zeros(data_augm_shape)
    evals_augm_train = np.zeros([n_samples * n_system, 2 * n_system])
    per_mat_x = np.zeros((n_spins, n_spins))
    per_mat_y = np.zeros((n_spins, n_spins))
    z = np.eye(n_spins)
    for i_site in range(n_spins):
        per_mat_x[i_site] = z[i_site - 1]
        per_mat_y[:, i_site] = z[:, i_site - 1]
    for i_sample in range(n_samples):
        data_set = input_data[i_sample]
        for i_x in range(n_spins):
            for i_y in range(n_spins):
                i_augm = i_sample * n_system + i_x * n_spins + i_y
                in_augm_train[i_augm] = data_set
                evals_augm_train[i_augm] = evals[i_sample]
                data_set = np.matmul(data_set, per_mat_y)
            data_set = np.matmul(per_mat_x, data_set)

    return in_augm_train, evals_augm_train


def apply_rotation(n_phi1, n_phi2, n_phi3, input_data, evals,
                   dimension, n_spins, n_samples):
    # TODO: Is this function independent of the system's dimensionality?
    """Apply data augmentation based on the continuous rotational symmetry of
    our Hamiltonian."""
    print("We are using the rotation invariance of the Hamiltonian to "
          "augment our data!")
    n_system = n_spins ** dimension
    phi1_list = np.linspace(0, 2 * np.pi, n_phi1, endpoint=False)
    phi2_list = np.linspace(0, np.pi, n_phi2, endpoint=True)
    phi3_list = np.linspace(0, 2 * np.pi, n_phi3, endpoint=False)
    n_angles = n_phi1 * n_phi2 * n_phi3 - 1
    data_augm_shape = [input_data.shape[i]
                       for i in range(len(input_data.shape))]
    data_augm_shape[0] *= n_angles
    in_rot_augm_train = np.zeros(data_augm_shape)
    evals_rot_augm_train = np.zeros([n_samples * n_angles, 2 * n_system])
    for i_sample in range(n_samples):
        dataset = input_data[i_sample]
        for i_phi3, phi3 in enumerate(phi3_list):
            for i_phi2, phi2 in enumerate(phi2_list):
                for i_phi1, phi1 in enumerate(phi1_list):
                    if phi1 or phi2 or phi3:
                        i_augm = n_angles * i_sample + \
                                 i_phi3 * n_phi2 * n_phi1 + \
                                 i_phi2 * n_phi1 + i_phi1 - 1
                        data_rotated = apply_euler_rotation(phi1, phi2, phi3,
                                                            dataset)
                        in_rot_augm_train[i_augm] = data_rotated
                        evals_rot_augm_train[i_augm] = evals[i_sample]

    return in_rot_augm_train, evals_rot_augm_train


def apply_euler_rotation(phi1, phi2, phi3, data):
    # TODO: Is this function independent of the system's dimensionality?
    rotation1 = np.array([[np.cos(phi1), -np.sin(phi1), 0],
                          [np.sin(phi1), np.cos(phi1), 0],
                          [0, 0, 1]])
    rotation2 = np.array([[np.cos(phi2), 0, np.sin(phi2)],
                          [0, 1, 0],
                          [-np.sin(phi2), 0, np.cos(phi2)]])
    rotation3 = np.array([[np.cos(phi3), -np.sin(phi3), 0],
                          [np.sin(phi3), np.cos(phi3), 0],
                          [0, 0, 1]])
    rotation_matrix = np.matmul(rotation3, np.matmul(rotation2, rotation1))
    data_shape = data.shape
    if len(data_shape) > 2:
        n_system = data.shape[1] ** (len(data.shape) - 1)
        data = data.reshape((3, n_system))
    data_rotated = np.matmul(rotation_matrix, data)
    if len(data_shape):
        data_rotated = data_rotated.reshape(data_shape)

    return data_rotated


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


def get_hkin(geometry, hopping=-1):
    """Determine the kinetic energy part of the Hamiltonian for 
       arbitrary geometry.
       geometry is a vector with the number of sites per dimension.
    """
    dimension = len(geometry)
    n_system = 1
    for i_dim in range(dimension):
        n_system *= geometry[i_dim]
    n_system = int(n_system)
    site_indices = np.arange(n_system)
    # TODO: Shouldn't this be twice the number lattice sites since we 
    #       have up and down fermions?
    kinetic = np.zeros((n_system, n_system), dtype=np.float64)
    # The way this code is written allows only hypercubic geometries. 
    # One might change the "length" variable and make it into an
    # array to accomodate hyper-rectangles.
    length = geometry[0]
    for site_index in site_indices:
        coords = find_coords(site_index, dimension, length)
        # TODO: Do we need the line below? Potential removal.
        # pos = coords * np.ones((dimension, dimension))
        for i_dim in range(dimension):
            coords_temp = np.copy(coords)
            # Hopping only to nearest neighbors.
            coords_temp[i_dim] += 1
            coords_temp[i_dim] %= length
            site_index_neg = find_site_index(coords_temp, length)
            kinetic[site_index, site_index_neg] = 1
            kinetic[site_index_neg, site_index] = 1
    kinetic *= hopping
    
    return csr_matrix(kinetic)


# TODO: Re-write the following code.
def construct_hamiltonian(thetas, phis, geometry, hopping, interaction):
    dimension = len(geometry)
    # THe line below is to initialize the n_system variable.
    n_system = 2 # The number two is because we have two species of fermions
                 # up and down.
    for i_dim in range(dimension):
        n_system *= geometry[i_dim]
    n_system = int(n_system)
    spin_up = n_system // 2
    site_indices = np.arange(spin_up)
    kinetic = np.zeros((spin_up, spin_up), dtype=np.float64)
    # The way this code is written allows only hypercubic geometries. 
    # One might change the "length" variable and make it into an
    # array to accomodate hyper-rectangles.
    length = geometry[0]
    for site_index in site_indices:
        coords = find_coords(site_index, dimension, length)
        # TODO: Do we need the line below? Potential removal.
        # pos = coords * np.ones((dimension, dimension))
        for i_dim in range(dimension):
            coords_temp = np.copy(coords)
            # Hopping only to nearest neighbors.
            coords_temp[i_dim] += 1
            coords_temp[i_dim] %= length
            site_index_neg = find_site_index(coords_temp, length)
            kinetic[site_index, site_index_neg] = 1
            kinetic[site_index_neg, site_index] = 1
    kinetic *= hopping
    hamiltonian = np.zeros((n_system, n_system), dtype=np.complex128)
    hamiltonian[:spin_up, :spin_up] = kinetic.copy()
    hamiltonian[spin_up:, spin_up:] = kinetic.copy()
    for site in range(spin_up):
        spin_z = 0.5 * np.cos(thetas[site])
        spin_p = 0.5 * np.exp(-1j * phis[site]) * np.sin(thetas[site])
        hamiltonian[site, site] = interaction * spin_z
        hamiltonian[site, site + spin_up] = interaction * spin_p
        hamiltonian[site + spin_up, site] = interaction * np.conj(spin_p)
        # The minus sign in the line below is to create the block for
        # the spin down.
        hamiltonian[site + spin_up, site + spin_up] = - interaction * spin_z

    return hamiltonian


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


def free_energy_exact(thetas, phis, beta, geometry, hopping, interaction,
                      chemical_potential, hamiltonian, hkin_data,
                      hkin_indices, hkin_indptr):
    hamiltonian = construct_hamiltonian(thetas, phis, geometry, hopping,
                                        interaction)
    evals, evecs = np.linalg.eigh(hamiltonian)
    free_energy = np.sum(np.logaddexp(0, -beta * (evals - chemical_potential),
                         dtype=np.float128))
    free_energy = -free_energy / beta

    return free_energy, (evals, evecs)


def sample_from_evals(thetas, phis, beta, n_spins, dimension, interaction,
                      chemical_potential, model):
    """Using neural networks in conjunction with Metropolis algorithm to
     generate samples without the need of exact diagonalization."""
    spin_comps = np.zeros(3 * n_spins ** dimension)
    spin_comps[0::3], spin_comps[1::3], spin_comps[2::3] = sph_to_cart(thetas,
                                                                       phis)
    spin_comps = interaction * torch.tensor(spin_comps, dtype=torch.float)
    evals = model(spin_comps)
    evals = evals.detach().numpy()
    free_energy = get_free_energy(beta, evals, chemical_potential)

    return free_energy, (evals,)


def sample_from_window(thetas, phis, beta, sampling_fun, sampling_args,
                       window, l_windows, dimension):
    """Using neural networks in conjunction with Metropolis algorithm to
     generate samples without the need of exact diagonalization. 
     l_windows define the windows used to evaluate the free energy of 
     the full system."""
    n_system = thetas.size
    # TODO: Improve the line below.
    n_spins = int(n_system ** (1 / dimension))
    # TODO: Make sure this works for 2D and up!
    if dimension > 1:
        window_ = (window for i_dim in range(dimension))
        window = window_
    n_windows = len(l_windows)
    l_free_energy = np.zeros(n_windows)
    if dimension == 1:
        l_evals = np.zeros((n_windows, 2 * window))
        for i_window, window_ in enumerate(l_windows):
            theta = thetas[window_: window_ + window]
            phi = phis[window_: window_ + window]
            f_, e_vals_vecs = sampling_fun(theta, phi, beta, *sampling_args)
            l_free_energy[i_window] = f_
            # TODO What's going on here? I think this is to make sure it 
            #   works with Exact models which also outputs the 
            #   eigenvectors.
            if len(e_vals_vecs):
                l_evals[i_window] = e_vals_vecs[0]
    elif dimension == 2:
        l_evals = np.zeros((n_windows, 2 * window[0] ** dimension))
        thetas = thetas.reshape((n_spins, n_spins))
        phis = phis.reshape((n_spins, n_spins))
        for i_window, window_ in enumerate(l_windows):
            theta = thetas[window_[0]: window_[0] + window[0],
                    window_[1]: window_[1] + window[1]]
            theta = theta.reshape(-1)
            phi = phis[window_[0]: window_[0] + window[0],
                  window_[1]: window_[1] + window[1]]
            phi = phi.reshape(-1)
            f_, e_vals_vecs = sampling_fun(theta, phi, beta, *sampling_args)
            l_free_energy[i_window] = f_
            # TODO What's going on here?
            if len(e_vals_vecs):
                l_evals[i_window] = e_vals_vecs[0]

    # TODO: Improve. Make sampling_args a dictionary in order to avoid 
    #   hardcoding the number 3 for assigning chemical_potential in the
    #   line below.
    chemical_potential = sampling_args[3]
    # TODO: Idea: Make a histogram and find the most probable free energy.
    # Using the mean of all the windows and then multiplying with the free 
    #   energy with the ratio of system size over number of sites in the 
    #   window give us the same answer.
    # free_energy = l_free_energy.mean()
    l_evals = l_evals.reshape(-1)
    free_energy = get_free_energy(beta, l_evals, chemical_potential)
    # TODO What's going on here?
    if not len(e_vals_vecs):
        window_args = ()
    else:
        # TODO This is needed to be consistent with the exact 
        # diagonalization function that outputs (E, V).
        l_evecs = ()
        window_args = (l_evals, l_evecs)

    return free_energy, window_args


def model_stats(evals, evals_pred, script_name, dim_str, n_spins, beta):
    rsq = r2_score(evals, evals_pred)
    mse = mean_squared_error(evals, evals_pred)
    mu_real = np.mean(evals)
    mu_pred = np.mean(evals_pred)
    var_real = np.cov(evals)
    var_pred = np.cov(evals_pred)
    fileName = f"docs/stats_{script_name}_{dim_str}_L_{n_spins}_beta_{beta}." \
               f"txt"
    with open(fileName, "w") as datafile:
        datafile.write("R^2\tMSE\tmu_real\tmu_pred\tvar_real\tvar_pred\n")
        datafile.write(f"{rsq:.4f}\t{mse:.3E}\t")
        datafile.write(f"{mu_real:.4f}\t{mu_pred:.4f}\t")
        datafile.write(f"{var_pred:.3E}\t{var_real:.3E}")
    print(f"R^2\t{rsq}\tMSE\t{mse:.3E}")
