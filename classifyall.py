#  Import any Python standard libraries you wish   #
# - I.e. libraries that do not require pip install with fresh
#   install of Python #

##################################
# ALLOWED NON-STANDARD LIBRARIES #
##################################
# Un-comment out the ones you use
import numpy as np
import pandas as pd
# import sklearn
# import tensorflow as tf
import torch as T
import torch.nn as nn
# import matplotlib
# import seaborn as sns
##################################

# NeuralNetwork Class
class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, p_dropout=0.20, save_dir="./trained_models"):
        super(NeuralNetwork, self).__init__()
        self.save_dir = save_dir

        activation = nn.ReLU()
        dropout = nn.Dropout(p=p_dropout)

        self.network = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=n_inputs * 3),
            activation,
            dropout,
            nn.Linear(in_features=n_inputs * 3, out_features=n_inputs * 2),
            activation,
            dropout,
            nn.Linear(in_features=n_inputs * 2, out_features=n_inputs),
            activation,
            dropout,
            nn.Linear(in_features=n_inputs, out_features=n_outputs),
        )
    
    def forward(self, X):
        logits = self.network(X)
        return logits

    def load(self, name):
        self.load_state_dict(T.load(f"{self.save_dir}/{name}.pth", map_location=T.device('cpu')))


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
    n_inputs = X.shape[1]
    n_outputs = 21 # 21 labels
    model = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs).to(device)

    model.load("NeuralNetwork-1_acc-51.14_loss-0.000004")

    # Classify
    # Change infer_labels - Currently just random
    # infer_labels = np.random.randint(0, 20, n_datapoints)
    
    model.eval()
    with T.no_grad():
        X = T.from_numpy(X).to(T.float32).to(device)
        infer_labels = model.forward(X).argmax(1)

    infer_labels = pd.DataFrame(infer_labels)

    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (n_datapoints, 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(n_datapoints, 1)}"

    infer_labels.to_csv("predlabels.txt", index=False, header=False)
    # infer_labels.to_csv("testlabels.txt", index=False, header=False)


if __name__ == "__main__":
    main()
