import torch
from torch.optim import SGD


def optimize(model, n_input, step=1e-2, grad_threshold=1e4, n_iter=int(1e3),
             bounds=[[1, -3], [10, 3]]):
    """
    Function for recommending parameter values to use in the subsequent
    simulation.  
    """
    # TODO: Find region where Hessian is zero.
    # Randomly initialize an input vector.
    vec_in = torch.rand(n_input)
    for i_iter in range(n_iter):
        print(f"i_iter {i_iter}")
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
                vec_in = torch.rand(n_input)
                i_iter = 0
                continue
            elif value > bounds[1][i_val]:
                vec_in = torch.rand(n_input)
                i_iter = 0
                continue

    return vec_in
