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

import torch as T
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, p_dropout=0.20, save_dir="./models"):
        super(NeuralNetwork, self).__init__()
        self.save_dir = save_dir

        activation = nn.ReLU()
        dropout = nn.AlphaDropout(p=p_dropout)

        self.network = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=1024),
            activation,
            dropout,
            nn.Linear(in_features=1024, out_features=512),
            activation,
            dropout,
            nn.Linear(in_features=512, out_features=256),
            activation,
            dropout,
            nn.Linear(in_features=256, out_features=n_outputs),
        )
    
    def forward(self, X):
        logits = self.network(X)
        return logits
    
    def save(self, name):
        T.save(self.state_dict(), f"{self.save_dir}/{name}.pth")

    def load(self, name):
        self.load_state_dict(T.load(f"{self.save_dir}/{name}.pth", map_location=T.device('cpu')))


# NeuralNetwork Ensemble Class
class Ensemble(nn.Module):
    def __init__(self, model_1, model_2, n_inputs, n_outputs, save_dir="./models"):
        super(Ensemble, self).__init__()
        self.save_dir = save_dir

        self.model_1 = model_1
        self.model_2 = model_2

        activation = nn.ReLU()

        self.classifier = nn.Sequential(
            activation,
            nn.Linear(in_features=n_inputs, out_features=n_outputs)
        )
    
    def forward(self, x):
        x_1 = self.model_1(x.clone())
        x_2 = self.model_2(x.clone())

        x = T.cat([x_1, x_2], dim=1)
        logits = self.classifier(x)
        return logits
    
    def save(self, name):
        T.save(self.state_dict(), f"{self.save_dir}/{name}.pth")

    def load(self, name):
        self.load_state_dict(T.load(f"{self.save_dir}/{name}.pth", map_location=T.device('cpu')))


device = T.device("cuda" if T.cuda.is_available() else "cpu")
model_1 = NeuralNetwork(n_inputs=63, n_outputs=21, p_dropout=0.4).to(device)
model_2 = NeuralNetwork(n_inputs=63, n_outputs=21, p_dropout=0.4).to(device)
model_1.save_dir = "./best_models"
model_2.save_dir = "./best_models"

#Freeze these models 
model_1.load("NeuralNetwork-1_acc-61.81_loss-0.000009")
for param in model_1.parameters():
    param.requires_grad_(False)

model_2.load("NeuralNetwork-2_acc-61.62_loss-0.000004")
for param in model_2.parameters():
    param.requires_grad_(False)


def convert_to_pca(components, mean, std, X):
    Z = (X - mean)/std
    return Z @ components.transpose()


def main():
    # TODO
    # pass
    test_data = pd.read_csv("testdata.txt", header=None)
    n_datapoints = test_data.shape[0]


    # Initialize prinicipal component analysis
    components = np.load("pca_utils/pca_components.npy")
    mean = np.load("pca_utils/X_mean.npy")
    std = np.load("pca_utils/X_std.npy")
    test_data = test_data.to_numpy()
    
    X = convert_to_pca(components, mean, std, test_data)

    # Initialize the model
    n_outputs = 21 # 21 labels
    model = Ensemble(model_1, model_2, n_outputs * 2, n_outputs).to(device)
    model.load("NeuralNetwork-Ensemble_acc-62.10_loss-0.005126")

    # Classify
    # Change infer_labels - Currently just random
    # infer_labels = np.random.randint(0, 20, n_datapoints)
    
    model.eval()
    with T.no_grad():
        X = T.from_numpy(X).to(T.float32).to(device)
        infer_labels = model.forward(X).argmax(1)

    infer_labels = pd.DataFrame(infer_labels.cpu())

    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (n_datapoints, 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(n_datapoints, 1)}"

    infer_labels.to_csv("predlabels.txt", index=False, header=False)
    # infer_labels.to_csv("testlabels.txt", index=False, header=False)


if __name__ == "__main__":
    main()
