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

# NeuralNetwork Class
class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, p_dropout=0.20, save_dir="./models"):
        super(NeuralNetwork, self).__init__()
        self.save_dir = save_dir

        activation = nn.ReLU()
        dropout = nn.AlphaDropout(p=p_dropout)

        self.network = nn.Sequential(
            nn.Flatten(),
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

    def load(self, name):
        self.load_state_dict(T.load(f"{self.save_dir}/{name}.pth", map_location=T.device('cpu')))


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

    # Initialize the model
    n_inputs = len(np.reshape(X[0], -1))
    n_outputs = 21 # 21 labels
    model = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs)

    model.load("NeuralNetwork-Image Dataset-2_acc-74.57_loss-0.000001")

    # Classify
    # Change infer_labels - Currently just random
    # infer_labels = np.random.randint(0, 20, n_datapoints)
    
    model.eval()
    with T.no_grad():
        X = T.from_numpy(X).to(T.float32)
        infer_labels = model.forward(X).argmax(1)

    infer_labels = pd.DataFrame(infer_labels)

    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (n_datapoints, 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(n_datapoints, 1)}"

    infer_labels.to_csv("predlabels.txt", index=False, header=False)
    # infer_labels.to_csv("testlabels.txt", index=False, header=False)


if __name__ == "__main__":
    main()
