import glob
import numpy as np
from config import data_dir_model, results_dir_model, site_indices
from utilities import calculate_absolute_magnetization
from utilities import calculate_absolute_staggered_magnetization
from utilities import calculate_quantity_stats


def save_quantity(filename, quant_name, quant_fn, quant_args,
                  results_dir_model):
    """
    Calculate quantity from specific data file
    """
    data = np.load(filename)
    quant_suffix = filename.split("data_")[1]
    d_pars = {}
    l_pars = quant_suffix.split("_")
    model_name = l_pars[0]
    # Skipping the first entry because it is the name.
    for i_par in range(1, len(l_pars) - 1, 2):
        if i_par == len(l_pars) - 2:
            l_pars[i_par + 1] = l_pars[i_par + 1].split(".npz")[0]
        d_pars[l_pars[i_par]] = l_pars[i_par + 1]

    quantity_mean, quantity_std = calculate_quantity_stats(data["samples"],
                                                           quant_fn,
                                                           quant_args
                                                           )
    quant_filename = f"{model_name}_quantity_{quant_name}_{quant_suffix}"
    d_pars["quantity_mean"] = quantity_mean
    d_pars["quantity_std"] = quantity_std
    np.savez(f"{results_dir_model}/{quant_filename}", **d_pars)

    return quantity_mean, quantity_std    


if __name__ == "__main__":
    d_quantities = dict(
        avg_magnetization=dict(
        quant_fn=calculate_absolute_magnetization, quant_args={}
        ),
        avg_stagg_magnetization=dict(
        quant_fn=calculate_absolute_staggered_magnetization,
        quant_args={"site_indices": site_indices}
        )
    )   
    
    for quant_name, d_quant in d_quantities.items():
        print(79 * "=")
        print(quant_name)
        print(39 * "+")
        # Load data.
        l_filenames = glob.glob(f"{data_dir_model}/data_*.npz")

        for filename in l_filenames:
            save_quantity(filename, quant_name, d_quant["quant_fn"],
                          d_quant["quant_args"], results_dir_model)
    print(79 * "=")
