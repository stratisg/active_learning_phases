import torch
from torch.optim import SGD


def optimize(model, n_input, step=1e-2, grad_threshold=1e4, n_iter=int(1e3)):
    """
    Function for recommending parameter values to use in the subsequent
    simulation.  
    """
    # Randomly initialize an input vector.
    vec_in = torch.rand(n_input)
    for i_iter in range(n_iter):
        vec_in.requires_grad = True
        grad = torch.autograd.grad(model(vec_in), vec_in)[0]
        if torch.norm(grad) > grad_threshold:
            break
        vec_in.requires_grad = False
        vec_in += grad * step

    return vec_in
