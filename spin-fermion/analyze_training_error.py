import numpy as np
from training_parameters import n_spins
from training_parameters import dimension
from training_parameters import n_train
from training_parameters import n_valid
from training_parameters import translation_toggle, rotation_toggle, data_augm


error_filename = f"docs/{data_augm}/training_error_sites_{n_spins}_" \
                 f"dimension_{dimension}_hopping_{hopping}_interaction_" \
                 f"{interaction}_train_{n_train}_valid_{n_valid}.npz"
