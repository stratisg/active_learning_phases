import numpy as np
from training_parameters import n_spins
from training_parameters import dimension
from training_parameters import n_phi1
from training_parameters import n_phi2
from training_parameters import n_phi3
from training_parameters import model_toggle
from utilities import apply_discrete_translation1d
from utilities import apply_discrete_translation2d
from utilities import apply_rotation
from utilities import appropriate_shape, flatten_data

def augment_dataset(in_data, evals, pars, data_augm_toggle, model_toggle):
    interaction = pars["interaction"]
    n_samples = len(in_data)
    data_augm_dict = dict(input_data=in_data, evals=evals,
                          dimension=dimension, n_spins=n_spins,
                          n_samples=n_samples)
    data_rot_augm_dict = dict(n_phi1=n_phi1, n_phi2=n_phi2, n_phi3=n_phi3, 
                              **data_augm_dict)
    # Translation invariance data augmentation
    translation_toggle = data_augm_toggle["translation_toggle"]
    rotation_toggle = data_augm_toggle["rotation_toggle"]
    if translation_toggle:
        if dimension == 1:
            input_augm_, evals_augm_ = \
                apply_discrete_translation1d(**data_augm_dict)
        elif dimension == 2:
            input_augm_, evals_augm_  = \
                apply_discrete_translation2d(**data_augm_dict)
        input_augm_ = flatten_data(input_augm_, n_spins, dimension)
    # Rotation invariance data augmentation
    if rotation_toggle:
        input_rot_augm_, evals_rot_augm_  = \
        apply_rotation(**data_rot_augm_dict)
        input_rot_augm_ = flatten_data(input_rot_augm_, n_spins, dimension)
    # Merge augmented data sets with initial training set.
    if translation_toggle and rotation_toggle:
        in_data = np.append(input_augm_, input_rot_augm_, axis=0)
        evals = np.append(evals_augm_, evals_rot_augm_, axis=0)
    elif rotation_toggle:
        in_data = flatten_data(in_data, n_spins, dimension)
        in_data = np.append(in_data, input_rot_augm_, axis=0)
        evals = np.append(evals, evals_rot_augm_, axis=0)
    elif translation_toggle:
        in_data = input_augm_
        evals = evals_augm_
    else:
        in_data = flatten_data(in_data, n_spins, dimension)  
    if model_toggle:
        l_interaction = np.array([interaction * np.ones(in_data.shape[0])]).T
        in_data = np.append(l_interaction, in_data, axis=1)
    else:
        in_data *= interaction
    print(f"Shape of data {in_data.shape}.")
    print(f"Shape of eigenvalues for data {evals.shape}.")
    
    return in_data, evals

