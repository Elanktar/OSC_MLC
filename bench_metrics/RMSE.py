import numpy as np
import sklearn.metrics as skm
import math

__all__ = ["RMSE"]


class RMSE:
    """Multi-label metrics using the hamming loss

    Calculates the hamming loss for each examples as the ratio of incorrect predicted labels divided by the total number of labels.
    """

    def __init__(self):
        self.sumRMSE = 0
        self.numberEvaluations = 0

    # Update function. Takes target and predicted label vectors and increments the accuracy sum
    def update(self, target: dict, predicted: dict):
        # Increment the evaluation counter
        self.numberEvaluations += 1

        # Ensure we have all the necessary data in predicted dict
        # to simplify evaluation loop
        targetKeys = target.keys()
        for k in targetKeys:
            if k not in predicted.keys():
                predicted[k] = 0

        # Prepare the counters
        pred_vector = []
        target_vector = []

        # Loop on predicted dictionary
        for k in predicted.keys():
            # Get values of the current label for predicted vector
            yPredicted = predicted[k]
            pred_vector.append(yPredicted)
            # Get values of the current label for target vector, defaulting to 0 if absent
            yTarget = target.get(k, 0)
            target_vector.append(yTarget)

            y_true = np.array(target_vector)
            y_pred = np.array(pred_vector)

            self.sumRMSE += math.sqrt(skm.mean_squared_error(y_true, y_pred))

    # Calculates and return the current average accuracy of evaluation for the metric
    def get(self):
        if self.numberEvaluations > 0:
            return self.sumRMSE / self.numberEvaluations
        else:
            print("Error, no evaluation conducted")
            return 0
