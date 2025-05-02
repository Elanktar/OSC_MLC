__all__ = ["NN_classifier"]

import numpy
import torch
from torch import nn
import re

from river import base

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


############################
#   NEURAL NETWORK MODEL   #
############################
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


#################################
#   NEURAL NETWORK CLASSIFIER   #
#################################


class NN_classifier(base.MultiLabelClassifier):
    def __init__(
        self,
        learning_rate=0.01,
        feature_size=1006,
        label_size=20,
    ):
        self.model = NeuralNetwork(feature_size, label_size)
        self.model = self.model.float()
        self.feature_size = feature_size
        self.label_size = label_size
        self.loss_fn = nn.BCELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def learn_one(self, features, labels):
        if type(labels) != dict:
            value = labels
            labels = dict()
            labels["L0"] = value
        new_x = numpy.zeros(self.feature_size)
        for key, value in features.items():
            features[key] = float(value)
            k = int(
                re.findall(r"\d+", key)[0]
            )  # "X49" -> "49". # = key pour le fichier test
            new_x[k] = value
        new_x = torch.tensor(new_x, dtype=torch.float32).to(self.device)
        new_y = numpy.zeros(self.label_size)
        for key, value in labels.items():
            k = int(re.findall(r"\d+", key)[0])
            new_y[k] = value
        new_y = torch.tensor(new_y, dtype=torch.float32).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(new_x)
        loss = self.loss_fn(outputs, new_y)
        loss.backward()
        self.optimizer.step()

    def predict_one(self, features):
        new_x = numpy.zeros(self.feature_size)
        for key, value in features.items():
            features[key] = float(value)
            k = int(
                re.findall(r"\d+", key)[0]
            )  # "X49" -> "49". # = key pour le fichier test
            new_x[k] = value
        new_x = torch.tensor(new_x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_tensor = self.model(new_x)
        pred_tensor = pred_tensor.cpu().numpy()
        n = 0
        dict_y_pred = dict()
        for j in pred_tensor:
            dict_y_pred["L{}".format(n)] = j
            n += 1
        if len(dict_y_pred) == 1:
            dict_y_pred = dict_y_pred["L0"]
        return dict_y_pred
