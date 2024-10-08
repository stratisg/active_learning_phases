import torch
from torch.optim import SGD


def generate_random_vector(n_input, bounds):
    """
    Generate random vector using uniform distribution given specific 
    bounds.
    """
    return torch.rand(n_input) * (bounds[1] - bounds[0]) + bounds[0]


def optimize(model, n_input, step=1e-2, grad_threshold=1e4, n_iter=int(1e3),
             bounds=[[1, -3], [10, 3]]):
    """
    Function for recommending parameter values to use in the subsequent
    simulation.  
    """
    # TODO: Find region where Hessian is zero.
    # Randomly initialize an input vector.
    bounds = torch.tensor(bounds, dtype=torch.float)
    vec_in = generate_random_vector(n_input, bounds)

    for i_iter in range(n_iter):
        vec_in.requires_grad = True
        grad = torch.autograd.grad(model(vec_in), vec_in)[0]
        if torch.norm(grad) > grad_threshold:
            break
        vec_in.requires_grad = False
        vec_in += grad * step
        # Check if the updated vector is within the bounds. If it is not
        #  it within bounds, it projects it within the bounded region.
        for i_val, value in enumerate(vec_in):
            if value < bounds[0][i_val]:
                print("Below low bound")
                vec_in = generate_random_vector(n_input, bounds)
            elif value > bounds[1][i_val]:
                print("Above high bound")
                vec_in = generate_random_vector(n_input, bounds)
    return vec_in
