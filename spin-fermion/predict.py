import time
import glob
import numpy as np
import torch
import os
from prediction_config import hopping, hopping_test
from prediction_config import interaction, interaction_test
from prediction_config import n_samples
from training_parameters import n_spins, n_train, n_valid, dimension
from training_parameters import hidden_1, loss
from training_parameters import translation_toggle, rotation_toggle
from training_parameters import model_toggle
from utilities import appropriate_shape
from plotting_functions import plot_pred_evals
from dataloading import augment_dataset


script_start = int(time.time())
pars = {}
n_system = n_spins ** dimension
pars["hopping"] = hopping_test
pars["interaction"] = interaction_test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"We are using {device} for this calculation!")
in_features = 3 * n_spins ** dimension
if model_toggle:
    in_features += 1
n_out = 2 * n_spins**dimension
pars["loss"] = loss
print("Parameters")
print(pars)
if loss == "MAE":
    loss_fn = torch.nn.L1Loss(reduction="mean")
elif loss == "MSE":
    loss_fn = torch.nn.MSELoss(reduction="mean")
else:
    print("Either choose MAE or MSE loss function!")
    quit()
print(f"data/*hopping_{hopping_test}_*interaction_{interaction_test}*")
datafile = glob.glob(f"data/*hopping_{hopping_test}_*" \
                     f"interaction_{interaction_test}*")[0]
print("We are using the datafile below as the test data set.")
print(datafile)
in_data, evals, evecs = appropriate_shape(datafile, dimension)
in_data, evals, _ = augment_dataset(in_data, evals, pars,
                                            dict(translation_toggle=False,
                                                 rotation_toggle=False),
                                            model_toggle)
in_data = torch.tensor(in_data, dtype=torch.float, device=device)
evals = torch.tensor(evals, dtype=torch.float, device=device)
model = torch.nn.Sequential(torch.nn.Linear(in_features=in_features, 
                                            out_features=hidden_1),
                            torch.nn.Tanh(), torch.nn.Linear(hidden_1, n_out))
if translation_toggle and rotation_toggle:
    data_augm = "trans_and_rotation_data_augm"
elif rotation_toggle:
    data_augm = "only_rotation_data_augm"
elif translation_toggle:
    data_augm = "only_translation_data_augm"
else:
    data_augm = "no_data_augm"
model_file = f"models/{data_augm}/sites_{n_spins}_dimension_{dimension}_" \
             f"hopping_{hopping}_interaction_{interaction}_train_{n_train}_" \
             f"valid_{n_valid}.pth"
model.load_state_dict(torch.load(model_file))
model.eval()
model.to(device)
evals_pred = model(in_data)
loss = loss_fn(evals_pred, evals)
print(f"Loss function: {loss:.2E}")
evals = evals.cpu().detach().numpy()
evals_pred = evals_pred.cpu().detach().numpy()
if not os.path.isdir("prediction"):
    os.mkdir("prediction")
os.chdir("prediction")
pred_dir = f"model_hopping_{hopping}_interaction_{interaction}"
if not os.path.isdir(pred_dir):
    os.mkdir(pred_dir)
os.chdir(pred_dir)
plot_pred_evals(evals, evals_pred, data_augm, n_spins,
                dimension, hopping_test, interaction_test, n_train, n_valid)
script_duration = int(time.time() - script_start)
print(f"Script duration: {script_duration} seconds!")

