import glob
import numpy as np
from config import data_dir_model, results_dir_model, site_indices
from utilities import calculate_absolute_magnetization
from utilities import calculate_absolute_staggered_magnetization
from utilities import calculate_quantity_stats


def get_quantity(filename, quant_name, quant_fn, quant_args,
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


def analyze_data(data_dir_model, results_dir_model, quant_name, quant_fn,
                 quant_args):
    """
    Generate quantities given the data generated with a given model.
    """
    # Load data.
    l_filenames = glob.glob(f"{data_dir_model}/data_*.npz")

    for i_file, filename in enumerate(l_filenames):
        print(f"Progress {(i_file + 1) / len(l_filenames):.3f}")
        _, _ = get_quantity(filename, quant_name, quant_fn, quant_args,
                            results_dir_model)

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
    analyze_data(data_dir_model, results_dir_model, quant_name, **d_quant)
print(79 * "=")
