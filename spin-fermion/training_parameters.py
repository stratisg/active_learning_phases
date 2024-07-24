n_spins = 20
n_samples = int(1e4)
dimension = 1
exact_type = "Exact"
p_train = 0.5
n_train = int(n_samples * p_train)
p_test = 0.25
n_test = int(n_samples * p_test)
n_valid = n_samples - (n_train + n_test)
hidden_1 = 80
hidden_2 = 10
n_epochs = int(1e3)
epoch_print = n_epochs // 10
d_epoch = n_epochs // 10
n_phi1 = 4
n_phi2 = 3
n_phi3 = 2
n_batch = 200
lr = 1e-3
gamma = 0.9995
weight_decay = 0
loss = "MAE"
make_figs = 1
translation_toggle = 1
rotation_toggle = 1
if translation_toggle and rotation_toggle:
    data_augm = "trans_and_rotation"
elif rotation_toggle:
    data_augm = "only_rotation_"
elif translation_toggle:
    data_augm = "only_translation"
else:
    data_augm = "no"
model_toggle = 0 
# In the above line we choose the model in the following way
# 0: Multiply inputs by interaction strength
# 1: Use interaction strength as an input.
data_toggle = "Exact" 
# In the above line we can choose the training data in the following way
# Exact: Uses data that were obtained for that specific beta.
# ExactUniform: Uses data that were generated using uniform distribution.
pre_training_toggle = 1
stop_criterion_toggle = 0
save_model_toggle = 1

