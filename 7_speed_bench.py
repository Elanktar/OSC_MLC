import pandas as pd
import numpy as np
import parameters
import os.path
import time
from codecarbon import OfflineEmissionsTracker
from river import metrics
from river import stream
from river import multioutput
from river import tree
from river import forest
from river import dummy
from river import stats
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import ParameterGrid
import implemented_models.NN
import implemented_models.NN_TL
import implemented_models.NN_TLH
import implemented_models.NN_TLH_fifo
import implemented_models.NN_TLH_sampling
import implemented_models.NN_TLH_memories
import implemented_models.NN_TLH_mini_memories
import implemented_models.NN_TLH_mini_memories_deep
import implemented_models.binary_relevance
import implemented_models.baseline_1_NN
import random
import argparse
import json
from copy import deepcopy

### Setting variables ########################################################

parser = argparse.ArgumentParser(description="Main")
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed, default to 1",
)
args = parser.parse_args()
seed = args.seed
bench = parameters.benchmark()
models = bench.models
workdir = bench.workdir


### Functions ###################################################################


def task_1_generator(rng_item):
    x = dict()
    y = dict()
    for i in range(100):
        x["{}".format(i)] = rng_item.random()
    for i in range(10):
        if i < 5 and x["{}".format(i)] >= 0.5:
            y["L{}".format(i)] = 1
        else:
            y["L{}".format(i)] = 0
    return [x, y]


def task_2_generator(rng_item):
    x = dict()
    y = dict()
    for i in range(100):
        x["{}".format(i)] = rng_item.random()
    for i in range(10):
        if i >= 5 and x["{}".format(i)] >= 0.5:
            y["L{}".format(i)] = 1
        else:
            y["L{}".format(i)] = 0
    return [x, y]


def online_eval(
    accuracy,
    y: dict,
    y_pred: dict,
    results: dict,
    task_signature,
):
    """Update the online metrics for the evaluation.

    Args:
        online_metrics_dict (dict): dict containing the online metrics.
        y (dict): dict containing the true label vector.
        y_pred (dict): predicted label vector
        metrics_online_results (dict): dict containing the online metrics results
    """
    for k_pred in y_pred.keys():
        if y_pred[k_pred] == None or float(y_pred[k_pred]) < 0.5:
            y_pred[k_pred] = 0
        elif float(y_pred[k_pred]) > 0.5:
            y_pred[k_pred] = 1
    for k_pred in y.keys():
        if k_pred not in y_pred:
            y_pred[k_pred] = 0
    for k_pred in y_pred.keys():
        if k_pred not in y:
            y[k_pred] = 0
    masked_y = dict()
    masked_y_pred = dict()
    for k_y in y.keys():
        if k_y in task_signature:
            masked_y[k_y] = y[k_y]
            masked_y_pred[k_y] = y_pred[k_y]
    accuracy.update(masked_y, masked_y_pred)
    print("Macro_BA : " + "{}".format(accuracy.get()))
    results["acc_results"].append(accuracy.get())


def first_exp_HPO(model_test, rng, cluster_label_signature):
    hpo_accuracy = metrics.multioutput.MacroAverage(metrics.BalancedAccuracy())

    # Initializing the code carbon tracker
    tracker = OfflineEmissionsTracker(
        tracking_mode="process",
        country_iso_code="FRA",
    )
    tracker.start()

    for i in range(1000):
        stream = task_1_generator(rng)
        x = stream[0]
        y = stream[1]
        y_pred = model_test.predict_one(x)
        for k_pred in y_pred.keys():
            if y_pred[k_pred] == None or float(y_pred[k_pred]) < 0.5:
                y_pred[k_pred] = 0
            elif float(y_pred[k_pred]) > 0.5:
                y_pred[k_pred] = 1
        for k_pred in y.keys():
            if k_pred not in y_pred:
                y_pred[k_pred] = 0
        for k_pred in y_pred.keys():
            if k_pred not in y:
                y[k_pred] = 0
        masked_y = dict()
        masked_y_pred = dict()
        for k_y in y.keys():
            if k_y in cluster_label_signature:
                masked_y[k_y] = y[k_y]
                masked_y_pred[k_y] = y_pred[k_y]
        hpo_accuracy.update(masked_y, masked_y_pred)
        # Training
        if (
            m == "NN_TL"
            or m == "NN_TLH"
            or m == "NN_TLH_fifo"
            or m == "NN_TLH_sampling"
            or m == "NN_TLH_memories"
            or m == "NN_TLH_mini_memories"
            or m == "NN_TLH_mini_memories_deep"
        ):
            model_test.learn_one(
                x,
                y,
                cluster_label_signature,
            )
        else:
            model_test.learn_one(x, y)
    tracker.stop()
    frugality = hpo_accuracy.get() - (1 / (1 + (1 / tracker._total_cpu_energy.kWh)))
    print("Accuracy : {}".format(hpo_accuracy.get()))
    print("Frugalité : {}".format(tracker._total_cpu_energy.kWh))
    return frugality


def save_bench_results(
    results: dict,
    m: str,
    workdir: str,
):
    """Save the metrics results in json files.

    Args:
        k_data (str): name of dataset
        m (str): name of model
        ordering (str): ordering used
        metrics_online_results (dict): dict containing the online metrics results
        metrics_continual_results (dict): dict containing the continual metrics values.
    """
    output_file = workdir + "results/Learning_speed_{}.json".format(m)
    with open(output_file, "w") as outfile:
        json.dump(results, outfile)


###########################################################################################
############### Main ######################################################################
###########################################################################################

rng = np.random.default_rng(seed)
task_1_signature = [
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
]
task_2_signature = [
    "L5",
    "L6",
    "L7",
    "L8",
    "L9",
]

for m in models:
    # Initializing the model
    models_parameter = {
        "NN": {"learning_rate": [0.1, 0.01, 0.001]},
        "NN_TL": {"learning_rate": [0.1, 0.01, 0.001]},
        "NN_TLH": {
            "learning_rate": [0.1, 0.01, 0.001],
            "hidden_sizes": [200, 2000],
        },
        "NN_TLH_fifo": {
            "learning_rate": [0.1, 0.01, 0.001],
            "hidden_sizes": [200, 2000],
            "size_replay_fifo": [100, 1000],
            "replay_fifo": [5, 10],
        },
        "NN_TLH_sampling": {
            "learning_rate": [0.1, 0.01, 0.001],
            "hidden_sizes": [200, 2000],
            "size_replay_sampling": [100, 1000],
            "replay_sampling": [5, 10],
        },
        "NN_TLH_memories": {
            "learning_rate": [0.1, 0.01, 0.001],
            "hidden_sizes": [200, 2000],
            "size_replay_fifo": [100, 1000],
            "replay_fifo": [5, 10],
            "size_replay_sampling": [100, 1000],
            "replay_sampling": [5, 10],
        },
        "NN_TLH_mini_memories": {
            "learning_rate": [0.1, 0.01, 0.001],
            "hidden_sizes": [200, 2000],
            "size_replay_fifo": [100, 1000],
            "replay_fifo": [5, 10],
            "size_replay_sampling": [100, 1000],
            "replay_sampling": [5, 10],
        },
        "NN_TLH_mini_memories_deep": {
            "learning_rate": [0.1, 0.01, 0.001],
            "size_replay_fifo": [100, 1000],
            "replay_fifo": [5, 10],
            "size_replay_sampling": [100, 1000],
            "replay_sampling": [5, 10],
            "weight_decay": [1e-4, 1e-5],
            "hidden_size": [128, 256, 512],
        },
        "BR_HT": {
            "grace_period": [100, 200],
            "delta": [1e-06, 1e-07],
            "tau": [0.05, 0.1],
        },
        "LC_HT": {
            "grace_period": [100, 200],
            "delta": [1e-06, 1e-07],
            "tau": [0.05, 0.1],
        },
        "CC_HT": {
            "grace_period": [100, 200],
            "delta": [1e-06, 1e-07],
            "tau": [0.05, 0.1],
        },
        "BR_random_forest": {
            "n_models": [5, 10, 15],
            "grace_period": [50, 100],
            "delta": [1e-06, 1e-07],
            "tau": [0.05, 0.1],
        },
        "iSOUPtree": {
            "grace_period": [100, 200],
            "delta": [1e-06, 1e-07],
            "tau": [0.05, 0.1],
        },
        "baseline_last_class": {},
        "baseline_prior_distribution": {},
        "baseline_mean": {},
        "baseline_oracle": {},
        "baseline_1_NN": {
            "equal_null_cos": [True, False],
            "eucli": [True, False],
            "max_window_size": [0, 10, 100, 1000],
        },
    }

    best_config = [0, 0, 0]
    list_config = ParameterGrid(models_parameter[m])
    nb_config = 10

    hpo_list = []

    # Bout de code pour créer une liste de config
    if nb_config < len(list_config):
        hpo_list = random.sample(range(len(list_config)), nb_config)

    else:
        for i in range(len(list_config)):
            hpo_list.append(i)

    for c in hpo_list:
        config = list_config[c]
        model_test = 0
        if m == "NN":
            model_test = implemented_models.NN.NN_classifier(
                learning_rate=config["learning_rate"],
                feature_size=100,
                label_size=10,
            )
        elif m == "NN_TL":
            model_test = implemented_models.NN_TL.NN_TL_classifier(
                learning_rate=config["learning_rate"],
                feature_size=100,
                label_size=10,
            )
        elif m == "NN_TLH":
            model_test = implemented_models.NN_TLH.NN_TLH_classifier(
                learning_rate=config["learning_rate"],
                feature_size=100,
                hidden_sizes=config["hidden_sizes"],
                label_size=10,
            )
        elif m == "NN_TLH_fifo":
            model_test = implemented_models.NN_TLH_fifo.NN_TLH_fifo_classifier(
                learning_rate=config["learning_rate"],
                feature_size=100,
                hidden_sizes=config["hidden_sizes"],
                label_size=10,
                size_replay_fifo=config["size_replay_fifo"],
                replay_fifo=config["replay_fifo"],
            )
        elif m == "NN_TLH_sampling":
            model_test = implemented_models.NN_TLH_sampling.NN_TLH_sampling_classifier(
                learning_rate=config["learning_rate"],
                feature_size=100,
                hidden_sizes=config["hidden_sizes"],
                label_size=10,
                size_replay_sampling=config["size_replay_sampling"],
                replay_sampling=config["replay_sampling"],
            )
        elif m == "NN_TLH_memories":
            model_test = implemented_models.NN_TLH_memories.NN_TLH_memories_classifier(
                learning_rate=config["learning_rate"],
                feature_size=100,
                hidden_sizes=config["hidden_sizes"],
                label_size=10,
                size_replay_fifo=config["size_replay_fifo"],
                replay_fifo=config["replay_fifo"],
                size_replay_sampling=config["size_replay_sampling"],
                replay_sampling=config["replay_sampling"],
            )
        elif m == "NN_TLH_mini_memories":
            model_test = (
                implemented_models.NN_TLH_mini_memories.NN_TLH_mini_memories_classifier(
                    learning_rate=config["learning_rate"],
                    feature_size=100,
                    hidden_sizes=config["hidden_sizes"],
                    label_size=10,
                    size_replay_fifo=config["size_replay_fifo"],
                    replay_fifo=config["replay_fifo"],
                    size_replay_sampling=config["size_replay_sampling"],
                    replay_sampling=config["replay_sampling"],
                )
            )
        elif m == "NN_TLH_mini_memories_deep":
            model_test = implemented_models.NN_TLH_mini_memories_deep.NN_TLH_mini_memories_deep_classifier(
                learning_rate=config["learning_rate"],
                feature_size=100,
                label_size=10,
                size_replay_fifo=config["size_replay_fifo"],
                replay_fifo=config["replay_fifo"],
                size_replay_sampling=config["size_replay_sampling"],
                replay_sampling=config["replay_sampling"],
                weight_decay=config["weight_decay"],
                hidden_size=config["hidden_size"],
            )
        elif m == "BR_HT":
            model_test = implemented_models.binary_relevance.binary_relevance(
                model=tree.HoeffdingTreeClassifier(
                    grace_period=config["grace_period"],
                    delta=config["delta"],
                    tau=config["tau"],
                )
            )
        elif m == "LC_HT":
            model_test = multioutput.MultiClassEncoder(
                model=tree.HoeffdingTreeClassifier(
                    grace_period=config["grace_period"],
                    delta=config["delta"],
                    tau=config["tau"],
                )
            )
        elif m == "CC_HT":
            model_test = multioutput.ClassifierChain(
                model=tree.HoeffdingTreeClassifier(
                    grace_period=config["grace_period"],
                    delta=config["delta"],
                    tau=config["tau"],
                )
            )
        elif m == "BR_random_forest":
            model_test = implemented_models.binary_relevance.binary_relevance(
                forest.ARFClassifier(
                    n_models=config["n_models"],
                    grace_period=config["grace_period"],
                    delta=config["delta"],
                    tau=config["tau"],
                )
            )
        elif m == "iSOUPtree":
            model_test = tree.iSOUPTreeRegressor(
                grace_period=config["grace_period"],
                delta=config["delta"],
                tau=config["tau"],
            )
        elif m == "baseline_1_NN":
            model_test = implemented_models.baseline_1_NN.baseline_1_NN(
                equal_null_cos=config["equal_null_cos"],
                eucli=config["eucli"],
                max_window_size=config["max_window_size"],
                label_size=10,
                feature_size=100,
            )
        if models_parameter[m] != {}:
            candidate_accuracy = first_exp_HPO(
                deepcopy(model_test), rng, task_1_signature
            )
            if best_config[0] < candidate_accuracy:
                best_config = [candidate_accuracy, config, deepcopy(model_test)]

    if m == "baseline_last_class":
        best_config = [
            0,
            0,
            implemented_models.binary_relevance.binary_relevance(
                dummy.NoChangeClassifier()
            ),
        ]
    elif m == "baseline_prior_distribution":
        best_config = [
            0,
            0,
            implemented_models.binary_relevance.binary_relevance(
                dummy.PriorClassifier()
            ),
        ]
    elif m == "baseline_mean":
        best_config = [
            0,
            0,
            implemented_models.binary_relevance.binary_relevance(
                dummy.StatisticRegressor(stats.Mean())
            ),
        ]

    with open(
        workdir + "Config/Speed_bench_{}.json".format(m),
        "w",
    ) as outfile:
        json.dump(best_config[1], outfile)

    macro_BA = metrics.multioutput.MacroAverage(metrics.BalancedAccuracy())
    results = {}
    results["acc_results"] = []
    results["count_1"] = 0
    results["count_2"] = 0

    model_final = []

    if m != "baseline_oracle":
        model_final = best_config[2]

    stream = task_1_generator(rng)
    x = stream[0]
    y = stream[1]
    # Test-then-train protocole :
    # Prediction :
    if m != "baseline_oracle":
        y_pred = model_final.predict_one(x)
    elif m == "baseline_oracle":
        y_pred = y
    # Online evaluation :
    online_eval(
        macro_BA,
        y,
        y_pred,
        results,
        task_1_signature,
    )
    # Training
    if (
        m == "NN_TL"
        or m == "NN_TLH"
        or m == "NN_TLH_fifo"
        or m == "NN_TLH_sampling"
        or m == "NN_TLH_memories"
        or m == "NN_TLH_mini_memories"
        or m == "NN_TLH_mini_memories_deep"
    ):
        model_final.learn_one(
            x,
            y,
            task_1_signature,
        )
    else:
        if m != "baseline_oracle":
            model_final.learn_one(x, y)
    results["count_1"] += 1

    while macro_BA.get() < 0.60:
        stream = task_1_generator(rng)
        x = stream[0]
        y = stream[1]
        # Test-then-train protocole :
        # Prediction :
        if m != "baseline_oracle":
            y_pred = model_final.predict_one(x)
        elif m == "baseline_oracle":
            y_pred = y
        # Online evaluation :
        online_eval(
            macro_BA,
            y,
            y_pred,
            results,
            task_1_signature,
        )
        # Training
        if (
            m == "NN_TL"
            or m == "NN_TLH"
            or m == "NN_TLH_fifo"
            or m == "NN_TLH_sampling"
            or m == "NN_TLH_memories"
            or m == "NN_TLH_mini_memories"
            or m == "NN_TLH_mini_memories_deep"
        ):
            model_final.learn_one(
                x,
                y,
                task_1_signature,
            )
        else:
            if m != "baseline_oracle":
                model_final.learn_one(x, y)
        results["count_1"] += 1

    while macro_BA.get() < 0.61:
        stream = task_2_generator(rng)
        x = stream[0]
        y = stream[1]
        # Test-then-train protocole :
        # Prediction :
        if m != "baseline_oracle":
            y_pred = model_final.predict_one(x)
        elif m == "baseline_oracle":
            y_pred = y
        # Online evaluation :
        online_eval(
            macro_BA,
            y,
            y_pred,
            results,
            task_1_signature,
        )
        # Training
        if (
            m == "NN_TL"
            or m == "NN_TLH"
            or m == "NN_TLH_fifo"
            or m == "NN_TLH_sampling"
            or m == "NN_TLH_memories"
            or m == "NN_TLH_mini_memories"
            or m == "NN_TLH_mini_memories_deep"
        ):
            model_final.learn_one(
                x,
                y,
                task_1_signature,
            )
        else:
            if m != "baseline_oracle":
                model_final.learn_one(x, y)
        results["count_2"] += 1

    # Saving results files
    save_bench_results(
        results,
        m,
        workdir,
    )
