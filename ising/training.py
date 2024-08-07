"""
Module with training functions.
"""
import torch


def training(data_in, data_out, model, loss_fn, optimizer, n_epochs=int(1e1),
             verbose=False):
    """
    Training function.
    """
    for i_epoch in range(n_epochs):
        for train_in, train_out in zip(data_in, data_out):
            prediction = model(train_in) 
            loss = loss_fn(prediction, train_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(79 * "=")
            prediction = model(data_in) 
            loss = loss_fn(prediction, data_out)
            print(f"Epoch {i_epoch:03d}\tLoss:{loss.item():.3e}")

    return model 
