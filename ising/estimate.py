import torch
from visualization import plot_quantity
from config import fit_model, model_dir_model, model_name, quant_name
from config import d_pars
from config import pics_dir_model, dpi


# Load model to estimate order parameters.
model_path = f"{model_dir_model}/{model_name}_{quant_name}.pth"
fit_model.load_state_dict(torch.load(model_path))
fit_model.eval()

l_values = []
# TODO: Avoid hardcoding

tensor_in = torch.zeros(len(d_pars.keys()))
for temp in d_pars["l_temperatures"]:
    tensor_in[0] = temp
    for interaction in d_pars["l_interactions"]:
        tensor_in[1] = interaction
        d_vals = {"temperature": temp, "interaction": interaction}
        estimate_out = fit_model(tensor_in).detach().numpy()
        d_vals["quantity_mean"] = estimate_out
        l_values.append(d_vals)

# Visualize estimate.
d_plot = dict(l_values=l_values, xlabel="J", ylabel="T")
figname = f"estimate_{model_name}_{quant_name}"
plot_quantity(quant_name, d_plot, figname, dpi, pics_dir_model)

# TODO: Generate error plot between model and ground truth.
# TODO: Add true phase boundaries and estimated phase boundaries.
