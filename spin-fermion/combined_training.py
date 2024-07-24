import os
import glob
import time
import numpy as np
import torch
from training_parameters import n_spins, dimension, n_samples
from training_parameters import n_train, n_valid, n_test
from training_parameters import hidden_1, n_batch, lr, gamma, weight_decay
from training_parameters import loss
from training_parameters import make_figs
from training_parameters import translation_toggle, rotation_toggle, data_augm
from training_parameters import model_toggle, data_toggle
from training_parameters import pre_training_toggle
from training_parameters import stop_criterion_toggle
from training_parameters import save_model_toggle
from generate_samples_config import hopping
from utilities import appropriate_shape
from plotting_functions import plot_pred_evals
from dataloading import augment_dataset
from training import training_loop


# TODO: Use data-loaders.
script_start = int(time.time())
pars = {}
n_system = n_spins ** dimension
data_augm_toggle = dict(translation_toggle=translation_toggle,
                        rotation_toggle=rotation_toggle)
# The input to our network is the spin components of each configuration 
# and the interaction strength.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"We are using {device} for this calculation!")

in_features = 3 * n_spins ** dimension
if model_toggle:
    in_features += 1
n_out = 2 * n_spins**dimension
pars["loss"] = loss
if loss == "MAE":
    loss_fn = torch.nn.L1Loss(reduction="mean")
elif loss == "MSE":
    loss_fn = torch.nn.MSELoss(reduction="mean")
else:
    print("Either choose MAE or MSE loss function!")
    quit()

data_dir = f"data/{data_toggle}"
data_header = f"{data_dir}/{data_toggle}_sites_{n_spins}_dimension_" \
              f"{dimension}_hopping_{hopping}_*samples_{n_samples}_*npz"
l_datafiles = glob.glob(data_header)
l_datafiles.sort()
print(f"l_datafiles {l_datafiles}")
pars["hopping"] = hopping
n_datafiles = len(l_datafiles)
for i_datafile, datafile in enumerate(l_datafiles):
    print(f"datafile {datafile}")
    interaction = float(datafile.split("interaction_")[1].split("_")[0])
    pars["interaction"] = interaction
    in_data, evals, evecs = appropriate_shape(datafile, dimension)
    shuffled_indices = np.random.default_rng().permutation(n_samples)
    train_indices = shuffled_indices[:n_train]
    in_train = in_data[train_indices]
    evals_train = evals[train_indices]
    # TODO: Need to fix the sample generation script for this line to work.
    #evecs_train = evecs[train_indices]
    valid_indices = shuffled_indices[n_train: n_train + n_valid]
    in_valid = in_data[valid_indices]
    evals_valid = evals[valid_indices]
    # TODO: Need to fix the sample generation script for this line to work.
    # evecs_valid = evecs[valid_indices]
    test_indices = shuffled_indices[n_train + n_valid:]
    in_test = in_data[test_indices]
    evals_test = evals[test_indices]
    # TODO: Need to fix the sample generation script for this line to work.
    # evecs_test = evecs[test_indices]
    in_train, evals_train = augment_dataset(in_train, evals_train, pars,
                                            data_augm_toggle, model_toggle)
    in_train = torch.tensor(in_train, dtype=torch.float, device=device)
    evals_train = torch.tensor(evals_train, dtype=torch.float, device=device)
    in_valid, evals_valid = augment_dataset(in_valid, evals_valid, pars,
                                            dict(translation_toggle=False, 
                                                 rotation_toggle=False),
                                             model_toggle)
    in_valid = torch.tensor(in_valid, dtype=torch.float, device=device)
    evals_valid = torch.tensor(evals_valid, dtype=torch.float, device=device)
    in_test, evals_test = augment_dataset(in_test, evals_test, pars,
                                          dict(translation_toggle=False, 
                                               rotation_toggle=False),
                                          model_toggle)
    in_test = torch.tensor(in_test, dtype=torch.float, device=device)
    evals_test = torch.tensor(evals_test, dtype=torch.float, device=device)
    # TODO: Create
    if not i_datafile:
        # in_total_shape = (n_samples * n_datafiles, n_system * 3)
        # out_total_shape = (n_samples * n_datafiles, n_system * 3)
        # in_train_total = torch.zeros(in_train.shape, dtype=torch.float,
        #                              device=device)
        in_train_total = in_train
        evals_train_total = evals_train
        in_valid_total = in_valid
        evals_valid_total = evals_valid
        in_test_total = in_test
        evals_test_total = evals_test
    else:
        in_train_total = torch.cat((in_train_total,in_train))
        evals_train_total = torch.cat((evals_train_total,evals_train))
        in_valid_total = torch.cat((in_valid_total,in_valid))
        evals_valid_total = torch.cat((evals_valid_total,evals_valid))
        in_test_total = torch.cat((in_test_total,in_test))
        evals_test_total = torch.cat((evals_test_total,evals_test))
print(f"in_train_total.shape {in_train_total.shape}")
print(f"in_valid_total.shape {in_valid_total.shape}")
print(f"in_test_total.shape {in_test_total.shape}")
batch_size = in_train.shape[0] // n_batch
model = torch.nn.Sequential(torch.nn.Linear(in_features=in_features, 
                                            out_features=hidden_1),
                            torch.nn.Tanh(), 
                            torch.nn.Linear(hidden_1, n_out))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
training_pars = dict(loss_fn=loss_fn, optimizer=optimizer,
                     scheduler=scheduler, batch_size=batch_size,
                     device=device)
interaction = "combined"
d_training_args = dict(hopping=hopping, interaction=interaction, model=model,
                       training_pars=training_pars, data_augm=data_augm,
                       stop_criterion_toggle=stop_criterion_toggle) 
trained_model = training_loop(in_train_total, evals_train_total,
                              in_valid_total, evals_valid_total,
                              **d_training_args)
trained_model.to(device)
evals_test_pred = trained_model(in_test_total)
loss = loss_fn(evals_test_pred, evals_test_total)
print(f"Prediction loss: {loss.item():.2E}")
evals_test_total = evals_test_total.cpu().detach().numpy()
evals_test_total  = evals_test_total.reshape(-1) 
evals_test_pred = evals_test_pred.cpu().detach().numpy()
evals_test_pred  = evals_test_pred.reshape(-1)
plot_pred_evals(evals_test_total, evals_test_pred, data_augm, n_spins,
                dimension, hopping, interaction, n_train, n_valid)
script_duration = int(time.time() - script_start)
print(f"Script duration: {script_duration} seconds!")

