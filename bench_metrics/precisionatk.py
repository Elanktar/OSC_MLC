import numpy as np

__all__ = ["precisionatk"]


class precisionatk:
    """Precisionatk for the k-top predictions.

    Count how many predicted labels are correct among the k-top labels (with the best confidence), then divide by k.
    """

    def __init__(self, k: int):
        self.k = k
        self.sumprecisionatk = 0
        self.numberEvaluations = 0

    # Update function. Takes target and predicted label vectors and increments the accuracy sum
    def update(self, target: dict, predicted: dict):
        # Increment the evaluation counter
        self.numberEvaluations += 1

        # Ensure we have all the necessary data in predicted dict
        # to simplify evaluation loop
        targetKeys = target.keys()
        for key in targetKeys:
            if key not in predicted.keys():
                predicted[key] = 0

        # Sort the predicted labels
        conf_abs = dict()
        for key, value in predicted.items():
            conf_abs[key] = max(1 - value, value)

        sorted_prediction = sorted(conf_abs.items(), key=lambda x: x[1], reverse=True)

        # Prepare the counters
        nb_correct_pred = 0

        for i in range(self.k):
            if predicted[sorted_prediction[i][0]] < 0.5:
                if target[sorted_prediction[i][0]] == 0:
                    nb_correct_pred += 1
            elif predicted[sorted_prediction[i][0]] >= 0.5:
                if target[sorted_prediction[i][0]] == 1:
                    nb_correct_pred += 1

        self.sumprecisionatk += nb_correct_pred / self.k

    # Calculates and return the current average accuracy of evaluation for the metric
    def get(self):
        if self.numberEvaluations > 0:
            return self.sumprecisionatk / self.numberEvaluations
        else:
            print("Error, no evaluation conducted")
            return 0
