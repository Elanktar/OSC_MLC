__all__ = ["NN_TLH_mini_memories_deep_classifier"]

import numpy
import torch
from torch import nn
import re
import random

from river import base

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


######################
#   RESIDUAL BLOCK   #
######################
# Quelles tailles pour input_size, hidden_size, output_size
class ResidualBlock(nn.Module):
    def __init__(self, io_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(io_size),
            nn.Linear(io_size, hidden_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_size, io_size),
        )

    def forward(self, x):
        return self.block(x) + x


##############
#   RESNET   #
##############
class ResNet10(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ResNet10, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            ResidualBlock(input_size, hidden_size),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


######################
#   REPLAY MEMORIES  #
######################
class FIFO:
    def __init__(self, size):
        self.size = size
        self.examples_seen = 0
        self.examples = []

    def add_example(self, features, labels, signature):
        self.examples_seen += 1
        if len(self.examples) < self.size:
            self.examples.append((features, labels, signature))
        else:
            index = self.examples_seen % self.size
            self.examples[index] = (features, labels, signature)

    def get_random_example(self):
        index = random.randint(0, len(self.examples) - 1)
        return self.examples[index]


class ReservoirSampling:
    def __init__(self, size):
        self.size = size
        self.examples_seen = 0
        self.examples = []

    def add_example(self, features, labels, signature):
        self.examples_seen += 1
        if len(self.examples) < self.size:
            self.examples.append((features, labels, signature))

        else:
            index = random.randint(0, self.examples_seen)
            if index < self.size:
                self.examples[index] = (features, labels, signature)

    def get_random_example(self):
        index = random.randint(0, len(self.examples) - 1)
        return self.examples[index]


#################################
#   NEURAL NETWORK CLASSIFIER   #
#################################
class NN_TLH_mini_memories_deep_classifier(base.MultiLabelClassifier):
    def __init__(
        self,
        learning_rate=0.01,
        feature_size=1006,
        hidden_size=256,
        size_replay_sampling=1000,
        replay_sampling=5,
        size_replay_fifo=1000,
        replay_fifo=5,
        label_size=20,
        weight_decay=1e-4,
    ):

        self.feature_size = feature_size
        self.label_size = label_size
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size

        # SETTING UP THE RESERVOIR SAMPLING REPLAY MEMORY
        self.sampling_memory = ReservoirSampling(size_replay_sampling)
        self.size_replay_sampling = size_replay_sampling
        self.replay_sampling = replay_sampling

        # SETTING UP THE FIFO REPLAY MEMORY
        self.FIFO_memory = FIFO(size_replay_fifo)
        self.size_replay_fifo = size_replay_fifo
        self.replay_fifo = replay_fifo

        self.model = ResNet10(self.feature_size, self.label_size, self.hidden_size)
        self.model = self.model.float()
        self.loss_fn = nn.BCELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # FUNCTION USED TO TRY TO STORE A NEW EXAMPLE INtO THE REPLAY MEMORIES (SAMPLING AND FIFO)
    def store_example(self, features, labels, signature):
        if self.size_replay_sampling > 0:
            self.sampling_memory.add_example(features, labels, signature)
        if self.size_replay_fifo > 0:
            self.FIFO_memory.add_example(features, labels, signature)

    def learn_one(self, features, labels, signature):
        # pre-traitement hors tensors GPU et gradient
        new_x = numpy.zeros(self.feature_size)
        for key, value in features.items():
            features[key] = float(value)
            k = int(
                re.findall(r"\d+", key)[0]
            )  # "X49" -> "49". # = key pour le fichier test
            new_x[k] = value
        new_y = numpy.zeros(self.label_size)
        for key, value in labels.items():
            if key in signature:
                k = int(re.findall(r"\d+", key)[0])
                new_y[k] = value

        mask = numpy.zeros(self.label_size)
        for i in range(self.label_size):
            if "L{}".format(i) in signature:
                mask[i] = 1.0
            else:
                mask[i] = 0

        features_lists = numpy.array(new_x, ndmin=2)
        labels_lists = numpy.array(new_y, ndmin=2)
        masks_lists = numpy.array(mask, ndmin=2)

        if (self.size_replay_sampling > 0) and (
            len(self.sampling_memory.examples) > self.replay_sampling
        ):
            for i in range(self.replay_sampling):
                past_example = self.sampling_memory.get_random_example()
                fArr = numpy.array(past_example[0], ndmin=2)
                lArr = numpy.array(past_example[1], ndmin=2)
                mArr = numpy.array(past_example[2], ndmin=2)
                features_lists = numpy.append(features_lists, fArr, axis=0)
                labels_lists = numpy.append(labels_lists, lArr, axis=0)
                masks_lists = numpy.append(masks_lists, mArr, axis=0)

        if (self.size_replay_fifo > 0) and (
            len(self.FIFO_memory.examples) > self.replay_fifo
        ):
            for i in range(self.replay_fifo):
                past_example = self.FIFO_memory.get_random_example()
                fArr = numpy.array(past_example[0], ndmin=2)
                lArr = numpy.array(past_example[1], ndmin=2)
                mArr = numpy.array(past_example[2], ndmin=2)
                features_lists = numpy.append(features_lists, fArr, axis=0)
                labels_lists = numpy.append(labels_lists, lArr, axis=0)
                masks_lists = numpy.append(masks_lists, mArr, axis=0)

        features_tensor = torch.from_numpy(features_lists).float().to(device)
        labels_tensor = torch.from_numpy(labels_lists).float().to(device)
        mask_tensor = torch.from_numpy(masks_lists).float().to(device)

        self.optimizer.zero_grad()
        outputs = self.model(features_tensor)
        masked_outputs = outputs * mask_tensor
        loss = self.loss_fn(masked_outputs, labels_tensor)
        loss.backward()
        self.optimizer.step()
        self.store_example(new_x, new_y, mask)

    def predict_one(self, features):
        new_x = numpy.zeros(self.feature_size)
        for key, value in features.items():
            features[key] = float(value)
            k = int(re.findall(r"\d+", key)[0])  # "X49" -> "49".
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
        return dict_y_pred
