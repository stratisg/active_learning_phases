import os
import numpy as np
import torch
from training_parameters import n_spins
from training_parameters import n_samples
from training_parameters import n_train
from training_parameters import n_valid
from training_parameters import dimension
from training_parameters import n_epochs
from training_parameters import epoch_print
from training_parameters import d_epoch
from training_parameters import make_figs
from plotting_functions import plot_train_error


def training_loop(in_train, evals_train, in_valid, evals_valid, hopping,
                  interaction, model, training_pars, data_augm="no_augm",
                  stop_criterion_toggle=False):
    """This is the training loop used to generate the neural network 
    models."""
    loss_fn = training_pars["loss_fn"]
    optimizer = training_pars["optimizer"]
    scheduler = training_pars["scheduler"]
    batch_size = training_pars["batch_size"]
    device = training_pars["device"]
    l_epoch = np.zeros([n_epochs, 2])
    epoch_opt = 0
    error_valid_opt = 1e1
    for epoch in range(n_epochs):
        shuffled_indices = torch.randperm(len(in_train))
        for i_ in range(len(in_train) // batch_size):
            ind_ = shuffled_indices[i_ * batch_size:(i_ + 1) * batch_size]
            in_batch = in_train[ind_]
            evals_batch = evals_train[ind_]
            evals_pred = model(in_batch)
            loss = loss_fn(evals_pred, evals_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        evals_pred = model(in_train)
        evals_pred_valid = model(in_valid)
        loss = loss_fn(evals_pred, evals_train)
        loss_valid = loss_fn(evals_pred_valid, evals_valid)
        err = loss.item()
        err_valid = loss_valid.item()
        l_epoch[epoch, 0] = err
        l_epoch[epoch, 1] = err_valid
        if not epoch % epoch_print:
            print(f"Epoch\t{epoch}")
            print(f"Train. error {err:.2E} Valid. error {err_valid:.2E}")
            model.to("cpu")
            if not os.path.isdir("models"):
                os.mkdir("models")
            if not os.path.isdir(f"models/{data_augm}"):
                os.mkdir(f"models/{data_augm}")
            model_check = f"models/{data_augm}/checkpoint_sites_{n_spins}_" \
                          f"dimension_{dimension}_hopping_{hopping}_" \
                          f"interaction_{interaction}_train_{n_train}_valid_" \
                          f"{n_valid}.pth"
            torch.save(model.state_dict(), model_check)
            model.to(device)
        if err_valid < error_valid_opt:
            epoch_opt = epoch
            err_train_opt = err
            error_valid_opt = err_valid
            model_opt = model
        if epoch >= 2 * d_epoch and stop_criterion_toggle:
            error0 = np.mean(l_epoch[epoch - 2 * d_epoch: epoch - d_epoch, 1])
            error1 = np.mean(l_epoch[epoch - d_epoch: epoch, 1])
            if error0 < error1:
                print(f"Error 0 {error0:.2E}\tError 1 {error1:.2E}")
                print(f"Stopping criterion achieved at epoch {epoch}!")
                break
    # TODO: Fix
    if not os.path.isdir("docs"):
        os.mkdir("docs/")
    if not os.path.isdir(f"docs/{data_augm}"):
        os.mkdir(f"docs/{data_augm}")
    error_filename = f"docs/{data_augm}/training_error_sites_{n_spins}_" \
                     f"dimension_{dimension}_hopping_{hopping}_interaction_" \
                     f"{interaction}_train_{n_train}_valid_{n_valid}.npz"
    np.savez(error_filename, training_error=l_epoch[:, 0],
             valid_error=l_epoch[:, 1])
    if make_figs:
        plot_train_error(n_epochs, l_epoch, epoch_opt, data_augm, n_spins,
                         dimension, hopping, interaction, n_train, n_valid)
    print(f"The last learning rate was {scheduler.get_last_lr()[0]:.2E}.")
    model = model_opt
    print(f"Epoch with optimum  error {epoch_opt}.")
    print(f"Train. opt error {err_train_opt:.2E}"
          f" Valid. opt. error {error_valid_opt:.2E}")
    model.to("cpu")
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not os.path.isdir(f"models/{data_augm}"):
        os.mkdir(f"models/{data_augm}")
    model_filename = f"models/{data_augm}/sites_{n_spins}_dimension_" \
                     f"{dimension}_hopping_{hopping}_interaction_" \
                     f"{interaction}_train_{n_train}_valid_{n_valid}.pth"
    torch.save(model.state_dict(), model_filename)
    os.remove(model_check )

    return model

