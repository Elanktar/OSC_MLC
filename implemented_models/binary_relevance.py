__all__ = ["binary_relevance"]

import numpy
import re
from river import base
from river import tree
from copy import deepcopy


class binary_relevance:
    def __init__(
        self,
        model=tree.HoeffdingTreeClassifier(),
    ):
        self.seen_labels = []
        self.models_ensemble = dict()
        self.model = model

    def learn_one(self, features, labels):
        for k_y, v_y in labels.items():
            if k_y in self.seen_labels:
                self.models_ensemble[k_y].learn_one(features, v_y)
            else:
                if v_y == 1 or v_y == -1:
                    self.seen_labels.append(k_y)
                    self.models_ensemble[k_y] = deepcopy(self.model)
                    self.models_ensemble[k_y].learn_one(features, v_y)

    def predict_one(self, features):
        output = dict()
        for k_model, v_model in self.models_ensemble.items():
            output[k_model] = self.models_ensemble[k_model].predict_one(features)
        return output
