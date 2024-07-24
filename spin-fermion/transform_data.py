import os
import  glob
import numpy as np
from generate_samples_config import dimension, hopping, interaction
from generate_samples_config import chemical_potential
from utilities import generate_filename


data_augm = ""
sites = 100
window = 100
samples = int(1e5)
train = 5000
valid = 2500
interaction_model = "combined"
modelname = "Exact"
data_dir = "data_transform/"
l_datafiles = glob.glob(f"{data_dir}/data_train_*")
for datafile in l_datafiles:
    data = np.load(datafile)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    f = data["f"]
    betas = data["beta"]
    beta = betas[0]
    evals = data["e_vals"]
    evecs = []
    filename = generate_filename(modelname, data_augm, sites, dimension,
                                 hopping, interaction, samples, beta,
                                 chemical_potential, window, train, valid,
                                 interaction_model)
    filename = data_dir + filename.split("_train")[0] + ".npz"
    np.savez(filename, beta=betas, x=x, y=y, z=z, f=f, evals=evals, evecs=evecs)

