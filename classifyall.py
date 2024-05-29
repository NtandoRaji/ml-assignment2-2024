#  Import any Python standard libraries you wish   #
# - I.e. libraries that do not require pip install with fresh
#   install of Python #

##################################
# ALLOWED NON-STANDARD LIBRARIES #
##################################
# Un-comment out the ones you use
import numpy as np
import pandas as pd
import sklearn
# import tensorflow as tf
import torch as T
import torch.nn as nn
from pickle import load
# import matplotlib
# import seaborn as sns
##################################


def convert_to_pca(components, mean, std, X):
    Z = (X - mean)/std
    return Z @ components.transpose()


def main():
    # TODO
    # pass
    test_data = pd.read_csv("testdata.txt", header=None)
    n_datapoints = test_data.shape[0]

    # Move a tensor to the GPU
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    # Initialize prinicipal component analysis
    components = np.load("pca_utils/pca_components.npy")
    mean = np.load("pca_utils/X_mean.npy")
    std = np.load("pca_utils/X_std.npy")
    test_data = test_data.to_numpy()
    
    X = convert_to_pca(components, mean, std, test_data)

    # Initialize the model
    with open("clf.pkl", "rb") as file:
        clf = load(file)

    # Classify
    # Change infer_labels - Currently just random
    # infer_labels = np.random.randint(0, 20, n_datapoints)
    infer_labels = clf.predict(X)

    infer_labels = pd.DataFrame(infer_labels)

    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (n_datapoints, 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(n_datapoints, 1)}"

    infer_labels.to_csv("predlabels.txt", index=False, header=False)
    # infer_labels.to_csv("testlabels.txt", index=False, header=False)


if __name__ == "__main__":
    main()

