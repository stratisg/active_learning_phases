import numpy as np
import numpy.random as rnd


def metropolis(n_samples, lattice, propose_fn, propose_args, sampling_fn,
               sampling_args, energy_fn, energ_args, temperature,
               k_boltzmann=1, warmup=int(1e3), seed_warmup=1821):
    """
    Metropolis algorithm for generating samples for a given lattice geometry.
    """
    # Lattice dimensionality
    n_dims = len(lattice)

    # Initialize configuration
    sample = sampling_fn(1, **sampling_args)
    n_sites = 1
    for sites_dim in lattice:
        n_sites *= sites_dim
    
    P_accept = 0
    P_attempt = 0
    rng_warmup = rnd.default_rng(seed_warmup)
    for i_warmup in range(warmup * n_sites):
        # Select a site at random to change its spin.
        site_proposed = np.zeros(n_dims)
        for i_dim, latice_dim in enumerate(lattice):
            site_proposed[i_dim] = rng_warmup.choice(latice_dim)
        
        # Store current spin in local variable
        spin_current = sample[site_proposed].copy()

        # Updated spin at proposed site.
        sample[site_proposed] = propose_fn(spin_current, **propose_args)

        # Calculate the energy of the updated system.

        # Calculate probability of configuration given the updated spin-config
        f_p, evals_vecs_p = f_func(thetas, phis, beta, *f_args)
        # calculate acceptance, if f_p <= f_c accept automatically
        # else use boltzmann weights
        accepted = (f_p <= f_c) or (np.random.uniform(0, 1) <
                                    np.exp(-beta * (f_p - f_c)))
        # update acceptance rate
        P_accept += accepted
        P_attempt += 1
        if accepted:  # if accepted do not need to change config just
            # update the current free energy of fermions
            f_c = f_p
            evals_vecs_c = evals_vecs_p
        else:  # if not accepted return the thetas and phis
            # back to original values
            thetas[i_p], phis[i_p] = theta_c, phi_c

    # Metropolis decorrelation length.
    dsample = int(P_attempt / P_accept)
    dsample = int(max(dsample, 1))
    accept_ratio = P_accept / P_attempt
    print(f"Acceptance ratio {accept_ratio:3f}.")
    print(f"Number of de-correlation samples {dsample * n_sites}.")
    n_montecarlo = n_samples * dsample
    for i_montecarlo in range(n_montecarlo):
        for i_site in range(n_sites):
            # randomly select site to change
            i_p = np.random.randint(n_sites)
            # store current spin in local variable
            theta_c, phi_c = thetas[i_p], phis[i_p]
            # propose an updated spin at site 'i_p'
            thetas[i_p] = np.arccos(np.random.uniform(-1, 1))
            phis[i_p] = np.random.uniform(-np.pi, np.pi)
            # calculate new free energy of fermions given this
            # updated spin-config
            f_p, evals_vecs_p = f_func(thetas, phis, beta, *f_args)
            # calculate acceptance, if f_p <= f_c accept automatically
            # else use boltzmann weights
            accepted = (f_p <= f_c) or (np.random.uniform(0, 1) <
                                        np.exp(-beta * (f_p - f_c)))
            # update acceptance rate
            P_accept += accepted
            P_attempt += 1
            if accepted:  # if accepted do not need to change config just
                # update the current free energy of fermions
                f_c = f_p
                evals_vecs_c = evals_vecs_p
            else:  # if not accepted return the thetas and phis
                # back to original values
                thetas[i_p], phis[i_p] = theta_c, phi_c
        # if number of monte-carlo steps reaches
        # multiple of the distance between samples
        # yield current data back to user to be
        # processed or stored
        accept_ratio = P_accept / P_attempt
        if i_montecarlo % dsample == 0:
            yield accept_ratio, f_c, thetas, phis, evals_vecs_c


def metropolis_adapt(n_sites, n_samples, beta, f_func, f_args, warmup=1000,
                     dsample=None, accept_ratio_opt=0.234, check_point=[]):
    script_start = time.time()
    time_out = 23 * 3600 + 57 * 60
    thetas = np.zeros(n_sites)
    phis = np.zeros(n_sites)
    # initialize configuration
    if len(check_point):
        print("Using check point!")
        data_file = check_point[0]
        data = np.load(data_file)
        x_check = data["x"][-1]
        y_check = data["y"][-1]
        z_check = data["z"][-1]
        f_check = data["f"][-1]
        evals_check = data["evals"][-1]
        thetas, phis = cart_to_sph(x_check, y_check, z_check)
    else:
        for i in range(n_sites):
            thetas[i] = np.arccos(np.random.uniform(-1, 1))
            phis[i] = np.random.uniform(-np.pi, np.pi)
    # calculate fermion free energy
    f_c, evals_vecs_c = f_func(thetas, phis, beta, *f_args)
    P_accept = 0
    P_attempt = 0
    # Initial S matrix is the same as the covariance matrix of a uniform
    #  distribution.
    s_mat = np.linalg.cholesky(np.eye(3)) / 3
    # s_mat = np.linalg.cholesky(np.eye(3))
    for m in range(warmup):
        if len(check_point):
            print("Skipped warmup stage since we use check point file!")
            break
        for n in range(n_sites):
            # randomly select site to change
            i_p = np.random.randint(n_sites)
            # store current spin in local variable
            theta_c, phi_c = thetas[i_p], phis[i_p]
            # propose an updated spin at site 'i_p'
            x, y, z = sph_to_cart(theta_c, phi_c)

            # Adaptive Metropolis based on Vihola 2012.
            vec = np.array([x, y, z])
            rnd_vec = np.random.default_rng().normal(size=(3,))
            vec += s_mat @ rnd_vec
            vec /= np.dot(vec, vec) ** 0.5
            rnd_vec = vec - np.array([x, y, z])
            x = vec[0]
            y = vec[1]
            z = vec[2]
            thetas[i_p], phis[i_p] = cart_to_sph_scalar(x, y, z)
            # calculate new free energy of fermions given this
            # updated spin-config
            f_p, evals_vecs_p = f_func(thetas, phis, beta, *f_args)
            # calculate acceptance, if f_p <= f_c accept automatically
            # else use boltzmann weights
            accepted = (f_p <= f_c) or (np.random.uniform(0, 1) <
                                        np.exp(-beta * (f_p - f_c)))
            accept_ratio_ = min(1, np.exp(-beta * (f_p - f_c)))
            eta = 1 / (m * n_sites + n + 1) ** 0.67
            rnd_mat = np.outer(rnd_vec, rnd_vec) / np.dot(rnd_vec, rnd_vec)
            mat_ = eta * (accept_ratio_ - accept_ratio_opt) * rnd_mat
            s_ = s_mat @ (np.eye(3) + mat_) @ s_mat.T
            s_mat = np.linalg.cholesky(s_)
            # End of adaptive step.

            # update acceptance rate
            P_accept += accepted
            P_attempt += 1
            if accepted:  # if accepted do not need to change config just
                # update the current free energy of fermions
                f_c = f_p
                evals_vecs_c = evals_vecs_p
            else:  # if not accepted return the thetas and phis
                # back to original values
                thetas[i_p], phis[i_p] = theta_c, phi_c
    if len(check_point):
        dsample = check_point[1]
        P_accept = check_point[2]
        P_attempt = check_point[3]
    else:
        if dsample is None:
            dsample = int(P_attempt / P_accept)
    dsample = int(max(dsample, 1))
    accept_ratio = P_accept / P_attempt
    print(f"Acceptance ratio {accept_ratio:3f}.")
    print(f"Number of de-correlation samples {dsample * n_sites}.")
    n_montecarlo = n_samples * dsample
    for i_montecarlo in range(n_montecarlo):
        for n in range(n_sites):
            # randomly select site to change
            i_p = np.random.randint(n_sites)
            # store current spin in local variable
            theta_c, phi_c = thetas[i_p], phis[i_p]
            # propose an updated spin at site 'i_p'
            x, y, z = sph_to_cart(theta_c, phi_c)

            # Adaptive Metropolis based on Vihola 2012.
            vec = np.array([x, y, z])
            rnd_vec = np.random.default_rng().normal(size=(3,))
            vec += s_mat @ rnd_vec
            vec /= np.dot(vec, vec) ** 0.5
            rnd_vec = vec - np.array([x, y, z])
            x = vec[0]
            y = vec[1]
            z = vec[2]
            thetas[i_p], phis[i_p] = cart_to_sph_scalar(x, y, z)
            # calculate new free energy of fermions given this
            # updated spin-config
            f_p, evals_vecs_p = f_func(thetas, phis, beta, *f_args)
            # calculate acceptance, if f_p <= f_c accept automatically
            # else use boltzmann weights
            accepted = (f_p <= f_c) or (np.random.uniform(0, 1) <
                                        np.exp(-beta * (f_p - f_c)))
            accept_ratio_ = min(1, np.exp(-beta * (f_p - f_c)))
            eta = 1 / (n + (warmup + m) * n_sites) ** 0.67
            mat_ = eta * (accept_ratio_ - accept_ratio_opt) \
                   * np.outer(rnd_vec, rnd_vec) / np.dot(rnd_vec, rnd_vec)
            s_ = s_mat @ (np.eye(3) + mat_) @ s_mat.T
            s_mat = np.linalg.cholesky(s_)
            # End of new Vihola 2012

            # update acceptance rate
            P_accept += accepted
            P_attempt += 1
            if accepted:  # if accepted do not need to change config just
                # update the current free energy of fermions
                f_c = f_p
                evals_vecs_c = evals_vecs_p
            else:  # if not accepted return the thetas and phis
                # back to original values
                thetas[i_p], phis[i_p] = theta_c, phi_c
        # if number of monte-carlo steps reaches
        # multiple of the distance between samples
        # yield current data back to user to be
        # processed or stored
        accept_ratio = P_accept / P_attempt
        if i_montecarlo % dsample == 0:
            yield accept_ratio, f_c, thetas, phis, evals_vecs_c


