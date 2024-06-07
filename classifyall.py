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
import sklearn.preprocessing
# import tensorflow as tf
import torch as T
import torch.nn as nn
# import matplotlib
# import seaborn as sns
##################################

import torch as T
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dims, n_outputs, p_dropout=0.20, save_dir="./models"):
        super(NeuralNetwork, self).__init__()
        self.save_dir = save_dir

        activation = nn.ReLU()
        dropout = nn.AlphaDropout(p=p_dropout)

        # Define layers with expected sizes
        self.network = nn.Sequential(
            nn.Conv2d(1, input_dims[0], kernel_size=3, padding=1),  # Input shape: (1, 32, 32)
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output shape: (32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output shape: (64, 16, 16)
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output shape: (64, 8, 8)
            
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=1024),
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

    def load(self, name):
        device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.load_state_dict(T.load(f"{self.save_dir}/{name}.pth", map_location=device))

# Ensemble Model
class Ensemble(nn.Module):
    def __init__(self, model_1, model_2, n_inputs, n_outputs, save_dir="./models"):
        super(Ensemble, self).__init__()
        self.save_dir = save_dir

        self.model_1 = model_1
        self.model_2 = model_2

        activation = nn.ReLU()
        dropout = nn.Dropout(p=0.4)

        self.classifier = nn.Sequential(
            activation,
            nn.Linear(in_features=n_inputs, out_features=256),
            activation,
            dropout,
            nn.Linear(in_features=256, out_features=128),
            activation,
            dropout,
            nn.Linear(in_features=128, out_features=n_outputs)
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
        device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.load_state_dict(T.load(f"{self.save_dir}/{name}.pth", map_location=device))


# Data Preprocessing Functions
def rotate_image(image, orientation):
    angle = 90 * orientation
    
    if angle == 0:
        return image
    elif angle == 90:
        return np.fliplr(np.transpose(image))  # Rotate 90 degrees clockwise
    elif angle == 180:
        return np.flipud(np.fliplr(image))  # Rotate 180 degrees
    elif angle == 270:
        return np.transpose(np.fliplr(image))  # Rotate 270 degrees clockwise


def preprocess_data(X):
    x = X[:-1]
    orientation = X[-1] # 4 orientations: 0, 1, 2, 3

    filtered_x = x[x >= 0] # Filter out negative values
    filtered_x = np.minimum(filtered_x, 255.0) # cap values greater than 255 to 255
    image = filtered_x.reshape([32, 32]) # reshape to an image

    normalized_image = sklearn.preprocessing.MinMaxScaler().fit_transform(image) # Normalize Image

    rotated_image = rotate_image(normalized_image, orientation) #Rotate Image
    return rotated_image


def main():
    # TODO
    # pass
    test_data = pd.read_csv("testdata.txt", header=None).to_numpy()
    n_datapoints = test_data.shape[0]
    
    # Preprocess Test Data
    preprocessed_test_data = []
    for data_point in test_data:
        preprocess_X = preprocess_data(data_point)
        preprocessed_test_data.append(preprocess_X)
    
    X = np.array(preprocessed_test_data)
    # Reshape X
    X = X.reshape([-1, 1, 32, 32])

    # Initialize the model
    n_inputs = [32, 32]
    n_outputs = 21 # 21 labels
    
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    model_1 = NeuralNetwork(input_dims=n_inputs, n_outputs=n_outputs, p_dropout=0.2).to(device)
    model_2 = NeuralNetwork(input_dims=n_inputs, n_outputs=n_outputs, p_dropout=0.2).to(device)

    #Freeze these models 
    model_1.load("Conv-NeuralNetwork-Image Dataset-1_acc-80.00_loss-0.000001")
    for param in model_1.parameters():
        param.requires_grad_(False)

    model_2.load("Conv-NeuralNetwork-Image Dataset-2_acc-80.19_loss-0.000001")
    for param in model_2.parameters():
        param.requires_grad_(False)

    model = Ensemble(model_1, model_2, 42, 21).to(device)

    model.load("Conv-NeuralNetwork-Ensemble-Advance_acc-80.67_loss-0.001625")

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
