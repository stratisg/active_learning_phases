import torch
from visualization import plot_quantity
from config import fit_model, model_dir_model, model_name, quant_name
from config import pics_dir_model, dpi


# Load model to estimate order parameters.
model_path = f"{model_dir_model}/{model_name}_{quant_name}.pth"
fit_model.load_state_dict(torch.load(model_path))
fit_model.eval()

# Generate input and output.
min_temp = 0
max_temp = 10
delta_temp = 1e-2
l_temps = torch.arange(min_temp, max_temp, delta_temp)

min_int = -3
max_int = 3
delta_int = 1e-2
l_interactions = torch.arange(min_int, max_int, delta_int)

l_values = []
# TODO: Avoid hardcoding
tensor_in = torch.zeros(2)
for temp in l_temps:
    tensor_in[0] = temp
    for interaction in l_interactions:
        tensor_in[1] = interaction
        d_pars = {"temperature": temp, "interaction": interaction}
        estimate_out = fit_model(tensor_in).detach().numpy()
        d_pars["quantity_mean"] = estimate_out
        l_values.append(d_pars)

# Visualize estimate.
d_plot = dict(l_values=l_values, xlabel="J", ylabel="T")
figname = f"estimate_{model_name}_{quant_name}"
plot_quantity(quant_name, d_plot, figname, dpi, pics_dir_model)
