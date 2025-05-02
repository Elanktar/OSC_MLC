import numpy

from river import base
import re

__all__ = ["baseline_1_NN"]


class baseline_1_NN(base.MultiLabelClassifier):
    def __init__(
        self,
        equal_null_cos=False,
        eucli=False,
        max_window_size=0,
        label_size=0,
        feature_size=0,
    ):
        self.equal_null_cos = equal_null_cos
        self.eucli = eucli
        self.label_size = label_size
        self.feature_size = feature_size

        # Setup window
        self.window = []
        self.max_window_size = max_window_size

    # Search function returning the index of the centroid closest in Feature space to the input feature vector
    def find_nearest_neighbour(self, x):
        if self.eucli == True:
            best_dist = float("inf")
            best_index = -1
            temp_x = numpy.zeros(self.feature_size)
            for f, v in x.items():
                temp_x[int(f)] = v
            x = temp_x

            for i in range(len(self.window)):
                dist = numpy.linalg.norm(x - self.window[i][0])
                if dist < best_dist:
                    best_dist = dist
                    best_index = i
            return best_index

        else:
            best_sim = -1
            best_index = -1
            temp_x = numpy.zeros(self.feature_size)
            for f, v in x.items():
                temp_x[int(f)] = v
            x = temp_x
            normFeatures = numpy.linalg.norm(x)
            for i in range(len(self.window)):
                normNeighbour = numpy.linalg.norm(self.window[i][0])
                if normFeatures == 0 and normNeighbour == 0:
                    sim = int(self.equal_null_cos)
                elif normFeatures == 0 or normNeighbour == 0:
                    sim = 0
                else:
                    sim = numpy.dot(x, self.window[i][0]) / (
                        normFeatures * normNeighbour
                    )
                if sim > best_sim:
                    best_sim = sim
                    best_index = i
            return best_index

    def learn_one(self, x, y):
        temp_x = numpy.zeros(self.feature_size)
        for f, v in x.items():
            temp_x[int(f)] = v
        x = temp_x
        self.window.append([x, y])
        if self.max_window_size != 0 and len(self.window) > self.max_window_size:
            self.window.pop(0)

    def predict_one(self, x):
        feature_neighbour_index = self.find_nearest_neighbour(x)

        if feature_neighbour_index == -1:
            print("Prediction error, model not trained")
            return {}

        # Extract and return the chosen centroid's labels in dense format
        return self.window[feature_neighbour_index][1]
