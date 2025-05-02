import pandas as pd
import argparse
import json
import parameters
import os.path
import time
from codecarbon import OfflineEmissionsTracker
from river import metrics
from river import stream
from river import multioutput
from river import tree
from river import forest
from river import preprocessing
from river import dummy
from river import stats
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import ParameterGrid
import bench_metrics.RMSE
import bench_metrics.precisionatk
import implemented_models.NN
import implemented_models.NN_TL
import implemented_models.NN_TLH
import implemented_models.NN_TLH_fifo
import implemented_models.NN_TLH_sampling
import implemented_models.NN_TLH_memories
import implemented_models.NN_TLH_mini_memories
import implemented_models.binary_relevance
import implemented_models.baseline_1_NN
import implemented_models.NN_TLH_mini_memories_deep
import random
from copy import deepcopy


### Setting variables ########################################################

parser = argparse.ArgumentParser(description="Main")
parser.add_argument(
    "--scenario",
    type=str,
    default="task_based",
    help="Other choice : task_free. Default to task_based.",
)
parser.add_argument(
    "--ordering",
    type=str,
    default="grad",
    help="grad for gradual drift, sudden for sudden drift, random for random ordering",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed, default to 1",
)
args = parser.parse_args()

scenario = args.scenario
ordering = args.ordering
seed = args.seed

bench = parameters.benchmark()
datasets = bench.datasets
models = bench.models
workdir = bench.workdir

### Functions ###################################################################


def get_ordering(k_data, workdir: str, scenario: str):
    """Get the ordering of the clusters in the stream according to the used ordering.

    Args:
        ordering (str): ordering used

    Returns:
        cluster_order (dict) : a dict containing the order of the clusters
    """
    assert os.path.isfile(
        workdir + "Orders/{}_random_".format(k_data) + "{}.json".format(scenario)
    ), "ordering not found in datasets directory"
    with open(
        workdir + "Orders/{}_random_".format(k_data) + "{}.json".format(scenario),
        "r",
        encoding="utf-8",
    ) as json_file:
        cluster_order = json.load(json_file)
    return cluster_order


def get_label_signature(k_data: str, workdir: str):
    """Get the label signature of each cluster.

    Args:
        k_data (str): name of the dataset

    Returns:
        cluster_label_signature (dict): dict containing the label signature of each label
    """
    assert os.path.isfile(
        workdir + "Labels/{}_cluster_labels_task_based.json".format(k_data)
    ), "label signature file not found in datasets directory"
    with open(
        workdir + "Labels/{}_cluster_labels_task_based.json".format(k_data),
        "r",
        encoding="utf-8",
    ) as json_file:
        cluster_label_signature = json.load(json_file)
    return cluster_label_signature


def get_task_based_data(
    cluster_order: dict,
    full_stream: dict,
    eval_sets: dict,
    k_data: str,
    workdir: str,
    scenario: str,
    ordering: str,
):
    """Get the data for a task based scenario.

    Args:
        cluster_order (dict): the dict containing the order of the clusters
        full_stream (dict): dict that will be filled with the full ordered data stream
        eval_sets (dict): the dict containing the eval sets for each cluster
        k_data (str): name of the dataset
    """
    seen_cluster = []
    stream_segmentation = dict()
    assert os.path.isfile(
        workdir + "Length/{}_".format(k_data) + "stream_length_{}.json".format(scenario)
    ), "length file not found"
    with open(
        workdir
        + "Length/{}_".format(k_data)
        + "stream_length_{}.json".format(scenario),
        "r",
        encoding="utf-8",
    ) as json_file:
        cluster_length = json.load(json_file)
    for k_order, v_order in cluster_order.items():
        if v_order not in seen_cluster:
            full_stream[k_order] = [
                v_order,
                pd.read_csv(
                    workdir
                    + "Experiences/{}".format(k_data)
                    + "_{}_".format(v_order)
                    + "exp_1.csv"
                ),
            ]
            data_eval = pd.read_csv(
                workdir
                + "Eval_set/{}".format(k_data)
                + "_{}_".format(v_order)
                + "eval_set.csv"
            )
            X_eval = data_eval.iloc[:, : (-1) * (v_data[0])]
            Y_eval = data_eval.iloc[:, (-1) * (v_data[0]) :]
            eval_sets[v_order] = [X_eval, Y_eval]
            seen_cluster.append(v_order)
            stream_segmentation[k_order] = [
                "cluster_{}_exp_1".format(v_order),
                cluster_length["cluster_{}_exp_1".format(v_order)],
            ]
        elif v_order in seen_cluster:
            full_stream[k_order] = [
                v_order,
                pd.read_csv(
                    workdir
                    + "Experiences/{}".format(k_data)
                    + "_{}_".format(v_order)
                    + "exp_2.csv"
                ),
            ]
            stream_segmentation[k_order] = [
                "cluster_{}_exp_2".format(v_order),
                cluster_length["cluster_{}_exp_2".format(v_order)],
            ]
    output_file = (
        workdir
        + "Length/{}_".format(k_data)
        + "stream_segmentation_{}_".format(scenario)
        + "{}.json".format(ordering)
    )
    with open(output_file, "w") as outfile:
        json.dump(stream_segmentation, outfile)


def online_eval(
    online_metrics_dict: dict,
    y: dict,
    y_pred: dict,
    metrics_online_results: dict,
    task: int,
):
    """Update the online metrics for the evaluation.

    Args:
        online_metrics_dict (dict): dict containing the online metrics.
        y (dict): dict containing the true label vector.
        y_pred (dict): predicted label vector
        metrics_online_results (dict): dict containing the online metrics results
    """
    for k in online_metrics_dict.keys():
        if k == "RMSE":
            for k_pred in y.keys():
                if k_pred not in y_pred:
                    y_pred[k_pred] = 0
            for k_pred in y_pred.keys():
                if k_pred not in y:
                    y[k_pred] = 0
                elif y_pred[k_pred] == None:
                    y_pred[k_pred] = 0
            online_metrics_dict[k].update(y, y_pred)
            print("{}".format(k) + " : " + "{}".format(online_metrics_dict[k].get()))
            metrics_online_results[k].append(online_metrics_dict[k].get())
        elif k == "precisionatk":
            for k_pred in y.keys():
                if k_pred not in y_pred:
                    y_pred[k_pred] = 0
            for k_pred in y_pred.keys():
                if k_pred not in y:
                    y[k_pred] = 0
                elif y_pred[k_pred] == None:
                    y_pred[k_pred] = 0
            masked_y = dict()
            masked_y_pred = dict()
            for k_y in y.keys():
                if k_y in cluster_label_signature["Cluster {}".format(task)]:
                    masked_y[k_y] = y[k_y]
                    masked_y_pred[k_y] = y_pred[k_y]
            online_metrics_dict[k].update(masked_y, masked_y_pred)
            print("{}".format(k) + " : " + "{}".format(online_metrics_dict[k].get()))
            metrics_online_results[k].append(online_metrics_dict[k].get())
        else:
            for k_pred in y_pred.keys():
                if y_pred[k_pred] < 0.5:
                    y_pred[k_pred] = 0
                else:
                    y_pred[k_pred] = 1
            online_metrics_dict[k].update(y, y_pred)
            print("{}".format(k) + " : " + "{}".format(online_metrics_dict[k].get()))
            metrics_online_results[k].append(online_metrics_dict[k].get())


def first_exp_HPO(model_test, full_stream, v_data, cluster_label_signature):
    hpo_accuracy = metrics.multioutput.MacroAverage(metrics.BalancedAccuracy())
    X_frame = full_stream["0"][1].iloc[:, : (-1) * (v_data[0])]
    Y_frame = full_stream["0"][1].iloc[:, (-1) * (v_data[0]) :]

    # Initializing the code carbon tracker
    tracker = OfflineEmissionsTracker(
        tracking_mode="process",
        country_iso_code="FRA",
    )
    tracker.start()

    for x, y in stream.iter_pandas(X_frame, Y_frame):
        y_pred = model_test.predict_one(x)
        # Online evaluation :
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
            if k_y in cluster_label_signature["Cluster {}".format(full_stream["0"][0])]:
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
                cluster_label_signature["Cluster {}".format(full_stream["0"][0])],
            )
        else:
            model_test.learn_one(x, y)
    tracker.stop()
    accuracy = hpo_accuracy.get()
    print("Accuracy : {}".format(hpo_accuracy.get()))
    print("Frugalité : {}".format(tracker._total_cpu_energy.kWh))
    return accuracy


def init_continual_metrics(
    eval_sets: dict,
    continual_metrics_dict: dict,
    metrics_continual_results: dict,
    iter_continual: int,
):
    """Initializes the continual metrics for each cluster on this round of continual evaluation.

    Args:
        eval_sets (dict): dict containing the evaluation sets data
        continual_metrics_dict (dict): dict containing the continual evaluation metrics
        metrics_continual_results (dict): dict that will be filled with metrics values.
        iter_continual (int): continual evaluation round number
    """
    for i in range(len(eval_sets)):
        continual_metrics_dict[
            "continual_macro_BA_cluster_{}_".format(i)
            + "round_{}".format(iter_continual)
        ] = metrics.multioutput.MacroAverage(metrics.BalancedAccuracy())
        metrics_continual_results[
            "continual_macro_BA_cluster_{}_".format(i)
            + "round_{}".format(iter_continual)
        ] = []


def continual_eval(
    continual_metrics_dict: dict,
    cl: int,
    iter_continual: int,
    y: dict,
    y_pred: dict,
):
    """Updates the metrics for continual evaluation.

    Args:
        continual_metrics_dict (dict): dict containing the continual evaluation metrics
        cl (int): number of cluster
        iter_continual (int): continual evaluation round number
        y (dict): dict containing the true label vector.
        y_pred (dict): predicted label vector
    """
    for k in y_pred.keys():
        if y_pred[k] == None or float(y_pred[k]) < 0.5:
            y_pred[k] = 0
        elif float(y_pred[k]) >= 0.5:
            y_pred[k] = 1
    for k in y.keys():
        if k not in y_pred:
            y_pred[k] = 0
    for k in y_pred.keys():
        if k not in y:
            y[k] = 0
    masked_y = dict()
    masked_y_pred = dict()
    for k_y in y.keys():
        if k_y in cluster_label_signature["Cluster {}".format(cl)]:
            masked_y[k_y] = y[k_y]
            masked_y_pred[k_y] = y_pred[k_y]
    continual_metrics_dict[
        "continual_macro_BA_cluster_{}_".format(cl) + "round_{}".format(iter_continual)
    ].update(masked_y, masked_y_pred)
    print(
        "continual_macro_BA_cluster_{}".format(cl)
        + " : "
        + "{}".format(
            continual_metrics_dict[
                "continual_macro_BA_cluster_{}_".format(cl)
                + "round_{}".format(iter_continual)
            ].get()
        )
    )


def continual_results_saving(
    metrics_continual_results: dict,
    continual_metrics_dict: dict,
    cl: int,
    iter_continual: int,
):
    """Save continual evaluation metrics results.

    Args:
        metrics_continual_results (dict): dict that is filled with metrics values.
        continual_metrics_dict (dict): dict containing the continual evaluation metrics
        cl (int): number of cluster
        iter_continual (int): continual evaluation round number
    """
    metrics_continual_results[
        "continual_macro_BA_cluster_{}_".format(cl) + "round_{}".format(iter_continual)
    ].append(
        continual_metrics_dict[
            "continual_macro_BA_cluster_{}_".format(cl)
            + "round_{}".format(iter_continual)
        ].get()
    )


def save_bench_results(
    k_data: str,
    m: str,
    ordering: str,
    metrics_online_results: dict,
    metrics_continual_results: dict,
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
    output_file = (
        workdir
        + "results/{}_".format(k_data)
        + "{}_".format(m)
        + "{}_online_".format(scenario)
        + "{}_results".format(ordering)
        + ".json"
    )
    with open(output_file, "w") as outfile:
        json.dump(metrics_online_results, outfile)
    output_file = (
        workdir
        + "results/{}_".format(k_data)
        + "{}_".format(m)
        + "{}_continual_".format(scenario)
        + "{}_results".format(ordering)
        + ".json"
    )
    with open(output_file, "w") as outfile:
        json.dump(metrics_continual_results, outfile)


###########################################################################################
############### Main ######################################################################
###########################################################################################

for k_data, v_data in datasets.items():

    # Getting ordering :
    cluster_order = get_ordering(k_data, workdir, scenario)

    # Getting label signature of each cluster :
    cluster_label_signature = get_label_signature(k_data, workdir)

    # Getting full_stream and eval_sets
    full_stream = dict()
    eval_sets = dict()
    get_task_based_data(
        cluster_order, full_stream, eval_sets, k_data, workdir, scenario, ordering
    )
    labels = []
    for i in range(v_data[0]):
        labels.append("y{}".format(i))

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
                    feature_size=v_data[1],
                    label_size=v_data[0],
                )
            elif m == "NN_TL":
                model_test = implemented_models.NN_TL.NN_TL_classifier(
                    learning_rate=config["learning_rate"],
                    feature_size=v_data[1],
                    label_size=v_data[0],
                )
            elif m == "NN_TLH":
                model_test = implemented_models.NN_TLH.NN_TLH_classifier(
                    learning_rate=config["learning_rate"],
                    feature_size=v_data[1],
                    hidden_sizes=config["hidden_sizes"],
                    label_size=v_data[0],
                )
            elif m == "NN_TLH_fifo":
                model_test = implemented_models.NN_TLH_fifo.NN_TLH_fifo_classifier(
                    learning_rate=config["learning_rate"],
                    feature_size=v_data[1],
                    hidden_sizes=config["hidden_sizes"],
                    label_size=v_data[0],
                    size_replay_fifo=config["size_replay_fifo"],
                    replay_fifo=config["replay_fifo"],
                )
            elif m == "NN_TLH_sampling":
                model_test = (
                    implemented_models.NN_TLH_sampling.NN_TLH_sampling_classifier(
                        learning_rate=config["learning_rate"],
                        feature_size=v_data[1],
                        hidden_sizes=config["hidden_sizes"],
                        label_size=v_data[0],
                        size_replay_sampling=config["size_replay_sampling"],
                        replay_sampling=config["replay_sampling"],
                    )
                )
            elif m == "NN_TLH_memories":
                model_test = (
                    implemented_models.NN_TLH_memories.NN_TLH_memories_classifier(
                        learning_rate=config["learning_rate"],
                        feature_size=v_data[1],
                        hidden_sizes=config["hidden_sizes"],
                        label_size=v_data[0],
                        size_replay_fifo=config["size_replay_fifo"],
                        replay_fifo=config["replay_fifo"],
                        size_replay_sampling=config["size_replay_sampling"],
                        replay_sampling=config["replay_sampling"],
                    )
                )
            elif m == "NN_TLH_mini_memories":
                model_test = implemented_models.NN_TLH_mini_memories.NN_TLH_mini_memories_classifier(
                    learning_rate=config["learning_rate"],
                    feature_size=v_data[1],
                    hidden_sizes=config["hidden_sizes"],
                    label_size=v_data[0],
                    size_replay_fifo=config["size_replay_fifo"],
                    replay_fifo=config["replay_fifo"],
                    size_replay_sampling=config["size_replay_sampling"],
                    replay_sampling=config["replay_sampling"],
                )
            elif m == "NN_TLH_mini_memories_deep":
                model_test = implemented_models.NN_TLH_mini_memories_deep.NN_TLH_mini_memories_deep_classifier(
                    learning_rate=config["learning_rate"],
                    feature_size=v_data[1],
                    label_size=v_data[0],
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
                    label_size=v_data[0],
                    feature_size=v_data[1],
                )
            if models_parameter[m] != {}:
                candidate_accuracy = first_exp_HPO(
                    deepcopy(model_test), full_stream, v_data, cluster_label_signature
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
            workdir
            + "Config/{}".format(k_data)
            + "_{}_".format(m)
            + "{}.json".format(scenario),
            "w",
        ) as outfile:
            json.dump(best_config[1], outfile)

        # Initializing the metrics
        min_task_size = 100000
        for k, v in full_stream.items():
            min_task_size = min(
                len(cluster_label_signature["Cluster {}".format(v[0])]),
                min_task_size,
            )
        nb_top = min(3, min_task_size)

        online_metrics_dict = dict()
        metrics_online_results = dict()

        online_metrics_dict = {
            "precisionatk": bench_metrics.precisionatk.precisionatk(nb_top),
            "RMSE": bench_metrics.RMSE.RMSE(),
            "macro_BA": metrics.multioutput.MacroAverage(metrics.BalancedAccuracy()),
        }
        metrics_online_results = {
            "precisionatk": [],
            "RMSE": [],
            "macro_BA": [],
        }

        metrics_continual_results = dict()
        continual_metrics_dict = dict()
        iter_continual = 0

        # Seen clusters and labels memory
        seen_clusters = []
        seen_labels = []

        # Initializing the code carbon tracker
        tracker = OfflineEmissionsTracker(
            tracking_mode="process",
            output_dir="consumption/",
            output_file="{}_".format(k_data)
            + "{}_".format(m)
            + "{}_".format(ordering)
            + "{}_".format(scenario)
            + "consumption"
            + ".csv",
            country_iso_code="FRA",
        )
        tracker.start()
        time_start = time.time()

        model_final = []

        if m != "baseline_oracle":
            model_final = best_config[2]

        # Benchmark
        for k_stream, v_stream in full_stream.items():
            if time.time() - time_start > 28800:
                break
            if v_stream[0] not in seen_clusters:
                seen_clusters.append(v_stream[0])
            X_frame = v_stream[1].iloc[:, : (-1) * (v_data[0])]
            Y_frame = v_stream[1].iloc[:, (-1) * (v_data[0]) :]
            for x, y in stream.iter_pandas(X_frame, Y_frame):
                if time.time() - time_start > 28800:
                    break
                # Test-then-train protocole :
                # Prediction :
                if m != "baseline_oracle":
                    y_pred = model_final.predict_one(x)
                elif m == "baseline_oracle":
                    y_pred = y
                # Online evaluation :
                online_eval(
                    online_metrics_dict, y, y_pred, metrics_online_results, v_stream[0]
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
                        cluster_label_signature["Cluster {}".format(v_stream[0])],
                    )
                else:
                    if m != "baseline_oracle":
                        model_final.learn_one(x, y)

            # Initializing the metrics for continual evaluation :
            init_continual_metrics(
                eval_sets,
                continual_metrics_dict,
                metrics_continual_results,
                iter_continual,
            )

            # Continual evaluation :
            for k_eval, v_eval in eval_sets.items():
                if time.time() - time_start > 28800:
                    break
                for x, y in stream.iter_pandas(
                    eval_sets[k_eval][0], eval_sets[k_eval][1]
                ):
                    if time.time() - time_start > 28800:
                        break
                    # Prediction
                    if m != "baseline_oracle":
                        y_pred = model_final.predict_one(x)
                    elif m == "baseline_oracle":
                        y_pred = y
                    # Evaluation
                    continual_eval(
                        continual_metrics_dict, k_eval, iter_continual, y, y_pred
                    )
                # Saving continual metrics results in a dict
                continual_results_saving(
                    metrics_continual_results,
                    continual_metrics_dict,
                    k_eval,
                    iter_continual,
                )
            iter_continual += 1
        # Stop code carbon tracker
        emissions: float = tracker.stop()

        # Saving results files
        save_bench_results(
            k_data,
            m,
            ordering,
            metrics_online_results,
            metrics_continual_results,
            workdir,
        )
