import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from generate_samples_config import n_spins, dimension, hopping, interaction
from generate_samples_config import n_samples


def make_hist(evals, datafile, n_spins, dimension, hopping, interaction,
              n_samples, figsize=(9, 6.5), color="blue", bins=20,
              label="Ground Truth", alpha=1):
    """Making a histogram of the  raw data."""
    plt.figure("Histogram", figsize=figsize)
    plt.xlabel("Eigenvalues", size=14)
    plt.ylabel("Frequency", size=14)
    plt.hist(evals.reshape(-1), color=color, bins=bins, density=True,
             label=label, alpha=alpha)
    plt.legend(fontsize=14)
    model_name = datafile.split("/")[1]
    filename = datafile.split("/")[2]
    if not os.path.isdir("pics"):
        os.mkdir("pics")
    model_path = f"pics/{model_name}"
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    pic_path = f"{model_path}/{model_name}_hist_sites_{n_spins}_" \
               f"dimension_{dimension}_hopping_{hopping}_interaction_" \
               f"{interaction}_samples_{n_samples}.png"    
    if model_name == "Exact":
        beta = filename.split("beta_")[1].split("_")[0]
        chemical_potential = filename.split("chemicalpotential_")[1]
        chemical_potential = chemical_potential.split("_")[0]
        add_str = f"_beta_{beta}_chemicalpotential_{chemical_potential}.png"
        pic_path = pic_path.replace(".png", add_str)
    plt.savefig(pic_path, dpi=600)
    plt.close()



def plot_train_error(n_epochs, l_error, epoch_opt, data_augm,
                     n_spins, dimension, hopping, interaction, n_train,
                     n_valid):
    """ Plotting the training and validation errors during the neural
     network's training. """
    plt.figure("Error v. Epoch")
    plt.semilogy(np.arange(n_epochs), l_error[:, 0], c="b",
                 label="Train error", alpha=0.6, marker="o")
    plt.semilogy(np.arange(n_epochs), l_error[:, 1], c="r",
                 label="Validation error", alpha=0.3, marker="s")
    plt.axvline(epoch_opt, c="g")
    plt.legend(fontsize=14)
    if not os.path.isdir("pics"):
        os.mkdir("pics")
    if not os.path.isdir(f"pics/{data_augm}"):
        os.mkdir(f"pics/{data_augm}")
    figname = f"pics/{data_augm}/error_sites_{n_spins}_dimension_" \
              f"{dimension}_hopping_{hopping}_interaction_{interaction}_" \
              f"train_{n_train}_valid_{n_valid}.png"
    plt.savefig(figname, dpi=600)
    plt.close()


def plot_pred_evals(evals, evals_pred, data_augm, n_spins, dimension, hopping,
                    interaction, n_train, n_valid, figsize=(9, 6.5), bins=20):
    """ Plotting the predicted free energy vs the real free energy. """

    plt.figure("Prediction", figsize=figsize)
    plt.xlabel("Real eigenvalues", size=14)
    plt.ylabel("Predicted eigenvalues", size=14)
    plt.scatter(evals.reshape(-1), evals_pred.reshape(-1), c="orange", s=36,
                marker="o", label="Prediction")
    plt.scatter(evals.reshape(-1), evals.reshape(-1), c="blue", marker="+",
                s=36, label="Real")
    plt.legend(fontsize=14)
    if not os.path.isdir("pics"):
        os.mkdir("pics")
    if not os.path.isdir(f"pics/{data_augm}"):
        os.mkdir(f"pics/{data_augm}")
    plt.savefig(f"pics/{data_augm}/prediction_sites_{n_spins}_dimension_" \
                f"{dimension}_hopping_{hopping}_interaction_{interaction}_" \
                f"train_{n_train}_valid_{n_valid}.png", dpi=600)
    plt.close()
    plt.figure("Histogram", figsize=figsize)
    plt.xlabel("Eigenvalues", size=14)
    plt.ylabel("Frequency", size=14)
    plt.hist(evals.reshape(-1), color="blue", bins=bins, density=True,
             label="True", alpha=0.5)
    plt.hist(evals_pred.reshape(-1), color="orange", bins=bins, density=True,
             alpha=0.3, label="Prediction")
    plt.legend(fontsize=14)
    plt.savefig(f"pics/{data_augm}/histogram_sites_{n_spins}_dimension_" \
                f"{dimension}_hopping_{hopping}_interaction_{interaction}_" \
                f"train_{n_train}_valid_{n_valid}.png", dpi=600)
    plt.close()

    
if __name__ == "__main__":
    script_start = int(time.time())
    l_models = ["Exact", "ExactUniform"]
    for model_name in l_models:
        print("====================")
        print(model_name)
        data_type = f"data/{model_name}/{model_name}_sites_{n_spins}_" \
                    f"dimension_{dimension}_hopping_{hopping}_interaction_" \
                    f"*{interaction}_samples_{n_samples}_*npz"
        if model_name == "ExactUniform":
            data_type = data_type.replace("_*npz", ".npz")
        l_datafiles = glob.glob(data_type)
        for datafile in l_datafiles:
            print(datafile)
            data = np.load(datafile)
            evals = data["evals"]
            make_hist(evals, datafile, n_spins, dimension, hopping,
                      interaction, n_samples)
    script_duration = int(time.time()) - script_start
    print(f"Script duration: {script_duration} seconds!")

