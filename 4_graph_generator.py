import matplotlib.pyplot as plt
import pandas as pd
import parameters
import argparse
import json
from os import path
import re

### Setting variables ########################################################

parser = argparse.ArgumentParser(description="Main")
parser.add_argument(
    "--scenario",
    type=str,
    default="task_based",
    help="1 if task_free scenario, default to 0",
)
parser.add_argument(
    "--ordering",
    type=str,
    default="grad",
    help="grad for gradual drift, sudden for sudden drift, random for random order",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="20NG",
    help="20NG, Bibtex, Birds, Bookmarks, Chess, Cooking, Corel16k001, Enron, Eukaryote, Human, Imdb, Mediamill, Ohsumed, Reuters-K500, Scene, Slashdot, tmc2007-500, Water-quality, Yeast, Yelp",
)
args = parser.parse_args()

scenario = args.scenario
ordering = args.ordering
data = args.dataset

bench = parameters.benchmark()
datasets = bench.datasets
models = bench.models
workdir = bench.workdir

color_list = [
    "deepskyblue",
    "steelblue",
    "aqua",
    "darkturquoise",
    "royalblue",
    "lightsteelblue",
    "dodgerblue",
    "limegreen",
    "green",
    "lightgreen",
    "yellowgreen",
    "olivedrab",
    "red",
    "salmon",
    "orangered",
    "tomato",
    "lightcoral",
    "magenta",
]

style_list = [
    "-",
    "--",
    "-.",
    ":",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    "-",
    "--",
    "-.",
    ":",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    "-",
    "--",
    "-.",
    ":",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    "-",
]

model_names = [
    "NN1",
    "NN2_TL",
    "NN3_TLH",
    "NN4_TLH_sampling",
    "NN5_TLH_fifo",
    "NN6_TLH_memories",
    "NN7_TLH_mini_memories",
    "NN8_TLH_mini_memories_deep",
    "T1_BR",
    "T2_LC",
    "T3_CC",
    "T4_ARF",
    "T5_SOUP",
    "B2_last_class",
    "B3_prior_distribution",
    "B4_mean",
    "B1_oracle",
    "B5_1NN",
]

### Functions ###################################################################


def set_online_results(
    workdir: str, data: str, model: str, ordering: str, scenario: str
):
    """Get the online metric names.

    Args:
        workdir (str): working directory
        data (str): name of the dataset
        model (str): name of the model used
        ordering (str): name of the ordering used (random, sudden, grad)
        scenario (str): task_free or task_based

    Returns:
        metrics(list): returns a list
    """
    metrics = []
    if path.isfile(
        workdir
        + "results/{}_".format(data)
        + "{}_".format(model)
        + "{}_online_".format(scenario)
        + "{}_results".format(ordering)
        + ".json"
    ):
        online = pd.read_json(
            workdir
            + "results/{}_".format(data)
            + "{}_".format(model)
            + "{}_online_".format(scenario)
            + "{}_results".format(ordering)
            + ".json"
        )
        for col_name in online.keys():
            metrics.append(col_name)
    return metrics


def get_length(workdir: str, data: str, scenario: str, ordering: str):
    """Get the length of each stream part.

    Args:
        workdir (str): working directory
        data (str): name of the dataset
        scenario (str): name of the scenario

    Returns:
        cluster_length(dict): dict containing the length of each stream part.
    """
    assert path.isfile(
        workdir
        + "Length/{}_".format(data)
        + "stream_segmentation_{}_".format(scenario)
        + "{}.json".format(ordering)
    ), "length file not found"
    with open(
        workdir
        + "Length/{}_".format(data)
        + "stream_segmentation_{}_".format(scenario)
        + "{}.json".format(ordering),
        "r",
        encoding="utf-8",
    ) as json_file:
        cluster_length = json.load(json_file)
    return cluster_length


def get_results(
    scenario: str, workdir: str, data: str, m: str, ordering: str, results: dict
):
    """Get the online metrics results.

    Args:
        scenario (str): scenario used
        workdir (str): working directory
        data (str): name of the dataset
        m (str): name of the model
        ordering (str): name of the stream ordering
        results (dict): dict that will contain the results

    Returns:
        step (list): list containing the results for the given parameters
    """
    if path.isfile(
        workdir
        + "results/{}_".format(data)
        + "{}_".format(m)
        + "{}_online_".format(scenario)
        + "{}_results".format(ordering)
        + ".json"
    ):
        new_data = pd.read_json(
            workdir
            + "results/{}_".format(data)
            + "{}_".format(m)
            + "{}_online_".format(scenario)
            + "{}_results".format(ordering)
            + ".json"
        )
        results[m] = new_data
        step = new_data.index.values.tolist()
        return step


def set_title(e: str, data: str, ordering: str, scenario: str):
    """Set the title of the graph

    Args:
        e (str): name of the metric
        data (str): name of the dataset
        ordering (str): name of the ordering used
        scenario (str): name of the scenario
    """
    plt.title("{} on ".format(e) + "{}".format(data))


def save_figure(
    data: str,
    ordering: str,
    scenario: str,
    e: str,
    workdir: str,
):
    """Save the figures.

    Args:
        data (str): name of the dataset
        ordering (str): name of the ordering
        scenario (str): name of the scenario
        e (str): name of the metric
        workdir (str): workind directory
    """
    plt.savefig(
        workdir
        + "graphs/{}_".format(data)
        + "{}_".format(ordering)
        + "{}_".format(scenario)
        + "{}_".format(e),
        bbox_inches="tight",
    )


###########################################################################################
############### Main ######################################################################
###########################################################################################

# Processing of a list of the used metrics :
metrics = set_online_results(workdir, data, models[0], ordering, scenario)
step = []
for e in metrics:
    plt.figure(figsize=(10, 5), dpi=600)
    cluster_length = get_length(workdir, data, scenario, ordering)
    current_length = 0
    for k_order, v_order in cluster_length.items():
        task = re.search(r"\d+", v_order[0]).group(0)
        current_length = current_length + v_order[1]
        plt.axvline(x=current_length, color="black", linestyle="-", linewidth=2)
        trans = plt.gca().get_xaxis_transform()
        plt.text(
            current_length - v_order[1] + 5,
            0,
            "Task {}".format(task),
            rotation=0,
            transform=trans,
        )
    results = dict()
    steps = []
    for m in models:
        steps.append(get_results(scenario, workdir, data, m, ordering, results))
    tab_res = dict()
    for k, v in results.items():
        tab_res[k] = v["macro_BA"].iloc[-1]
    output_file = workdir + "results/{}_".format(data) + "tab_online_{}.json".format(e)
    with open(output_file, "w") as outfile:
        json.dump(tab_res, outfile, default=str)
    result_array = []
    for k, v in results.items():
        results[k] = v[e].to_list()
        result_array.append(results[k])
    step_max = 0
    for j in range(len(result_array)):
        if step_max < len(steps[j]):
            step_max = len(steps[j])
        plt.plot(
            steps[j],
            result_array[j],
            color=color_list[j],
            label=model_names[j],
            linestyle=style_list[j],
        )
    plt.xlim(0, step_max)
    # plt.ylim(bottom=0)
    plt.ylabel(e)
    plt.xlabel("Instances")
    plt.grid(True)
    plt.legend(
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        ncol=1,
        fancybox=True,
        shadow=True,
    )
    set_title(e, data, ordering, scenario)
    plt.tight_layout()
    save_figure(data, ordering, scenario, e, workdir)
    plt.close()
