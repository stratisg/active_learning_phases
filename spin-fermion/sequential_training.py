import os
import glob
import time
import numpy as np
import torch
from training_parameters import n_spins, n_samples, dimension
from training_parameters import n_train, n_valid, n_test
from training_parameters import data_toggle
from training_parameters import hidden_1
from training_parameters import n_batch
from training_parameters import lr
from training_parameters import gamma
from training_parameters import weight_decay
from training_parameters import loss
from training_parameters import make_figs
from training_parameters import translation_toggle
from training_parameters import rotation_toggle
from training_parameters import data_augm
from training_parameters import model_toggle
from training_parameters import pre_training_toggle
from training_parameters import stop_criterion_toggle
from training_parameters import save_model_toggle
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
# TODO: What do we do for the multiple beta, chemical potential, 
#   and interaction strength?
data_type = f"data/{data_toggle}/{data_toggle}_sites_{n_spins}_dimension_" \
            f"{dimension}_hopping_{hopping}*samples_{n_samples}*npz"
l_datafiles = glob.glob()
l_datafiles.sort()
quit()
print(f"l_datafiles {l_datafiles}")
for i_datafile, datafile in enumerate(l_datafiles):
    print(f"datafile {datafile}")
    hopping = float(datafile.split("hopping_")[1].split("_")[0])
    pars["hopping"] = hopping
    interaction = float(datafile.split("interaction_")[1].split("_")[0])
    pars["interaction"] = interaction
    in_data, evals, evecs = appropriate_shape(datafile, dimension)
    shuffled_indices = np.random.default_rng().permutation(n_samples)
    train_indices = shuffled_indices[:n_train]
    in_train = in_data[train_indices]
    evals_train = evals[train_indices]
    # evecs_train = evecs[train_indices]
    valid_indices = shuffled_indices[n_train: n_train + n_valid]
    in_valid = in_data[valid_indices]
    evals_valid = evals[valid_indices]
    # evecs_valid = evecs[valid_indices]
    test_indices = shuffled_indices[n_train + n_valid:]
    in_test = in_data[test_indices]
    evals_test = evals[test_indices]
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
    
    batch_size = in_train.shape[0] // n_batch
    if pre_training_toggle and i_datafile:
        print("Using pre-training!")
        model = trained_model
    else:
        print("No pre-training!")
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
    trained_model = training_loop(in_train, evals_train, in_valid, evals_valid,
                                  hopping, interaction, model, training_pars,
                                  data_augm, stop_criterion_toggle)
    in_test, evals_test = augment_dataset(in_test, evals_test, pars,
                                          dict(translation_toggle=False, 
                                               rotation_toggle=False),
                                          model_toggle)
    trained_model.to(device)
    in_test = torch.tensor(in_test, dtype=torch.float, device=device)
    evals_test = torch.tensor(evals_test, dtype=torch.float, device=device)
    evals_test_pred = trained_model(in_test)
    loss = loss_fn(evals_test_pred, evals_test)
    print(f"Prediction loss: {loss.item():.2E}")
    evals_test = evals_test.cpu().detach().numpy()
    evals_test_pred = evals_test_pred.cpu().detach().numpy()
    plot_pred_evals(evals_test, evals_test_pred, data_augm, n_spins,
                    dimension, hopping, interaction, n_train, n_valid)
script_duration = int(time.time() - script_start)
print(f"Script duration: {script_duration} seconds!")

