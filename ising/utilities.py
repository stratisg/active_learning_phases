import numpy as np


def calculate_magnetization(spin_config):
    """
    Calculate the system's magnetization per site.
    """ 

    return spin_config.mean()


def calculate_absolute_magnetization(spin_config):
    """
    Calculate the absolute value of the system's magnetization per site.
    """

    return np.abs(calculate_magnetization(spin_config))


def calculate_absolute_staggered_magnetization(spin_config, site_indices):
    """
    Calculate the absolute value of the staggered magnetization per site.
    """
    template = np.ones_like(spin_config)
    for site_index in site_indices:
        template[tuple(site_index)] =  (-1) ** site_index.sum()

    spin_config = np.multiply(spin_config, template)
    
    return calculate_absolute_magnetization(spin_config)


def calculate_quantity_stats(l_samples, quantity_fn, quant_args):
    """
    Calculate the sample mean and sample standard deviation for the 
    given quantity. 
    """
    quantity = np.zeros(len(l_samples))
    # TODO: Consider using map().
    for i_sample, sample in enumerate(l_samples):
        quantity[i_sample] = quantity_fn(sample, **quant_args)

    return quantity.mean(), quantity.std()
