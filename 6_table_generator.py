import pandas as pd
import argparse
import json
import parameters
import os.path
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import scikit_posthocs as sp
import utils.utils as utils
from adjustText import adjust_text

### Setting variables ########################################################

parser = argparse.ArgumentParser(description="Main")
parser.add_argument(
    "--scenario",
    type=str,
    default="task_based",
    help="Other choice : task_free. Default to task_based.",
)
parser.add_argument(
    "--time_stamped",
    type=bool,
    default=False,
    help="Wether the dataset is initially ordered by a timestamp or not. Default to False.",
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
time_stamped = args.time_stamped
seed = args.seed

bench = parameters.benchmark()
datasets = bench.datasets
models = bench.models
workdir = bench.workdir

metrics_continual = [
    "average_accuracy",
    "neg_bwt",
    "pos_bwt",
    "neg_fwt",
    "pos_fwt",
    "frugality_score",
]

metrics_consumption = [
    "duration",
    "cpu_energy",
    "ram_energy",
    "energy_consumed",
]

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

model_names = {
    "NN": "NN1",
    "NN_TL": "NN2_TL",
    "NN_TLH": "NN3_TLH",
    "NN_TLH_sampling": "NN4_TLH_sampling",
    "NN_TLH_fifo": "NN5_TLH_fifo",
    "NN_TLH_memories": "NN6_TLH_memories",
    "NN_TLH_mini_memories": "NN7_TLH_mini_memories",
    "NN_TLH_mini_memories_deep": "NN8_TLH_mini_memories_deep",
    "BR_HT": "T1_BR",
    "LC_HT": "T2_LC",
    "CC_HT": "T3_CC",
    "BR_random_forest": "T4_ARF",
    "iSOUPtree": "T5_SOUP",
    "baseline_last_class": "B2_last_class",
    "baseline_prior_distribution": "B3_prior_distribution",
    "baseline_mean": "B4_mean",
    "baseline_oracle": "B1_oracle",
    "baseline_1_NN": "B5_1NN",
}

model_names_list = [
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


metrics_online = ["macro_BA", "precisionatk", "RMSE"]

coord_consumption = {}
coord_time = {}

# pour les métriques online :
for metric in metrics_online:
    metric_dict = {}
    intermediary_res = {}
    for m in models:
        intermediary_res[m] = []
    for set_data in datasets:
        with open(
            workdir
            + "results/{}_tab_online_".format(set_data)
            + "{}.json".format(metric),
            "r",
            encoding="utf-8",
        ) as json_file:
            file = json.load(json_file)
        for tested_model in models:
            intermediary_res[tested_model].append(file[tested_model])

    for m in models:
        sum = 0
        metric_count = 0
        for i in intermediary_res[m]:
            intermediary_res[m][metric_count] = float(i)
            sum += float(i)
            metric_count += 1
        if metric_count != 0:
            mean = sum / metric_count
        else:
            mean = 0
        intermediary_res[m].append(mean)

    metric_dict = intermediary_res
    metric_df = pd.DataFrame(metric_dict)
    metric_df.insert(
        0,
        "Datasets",
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
            "Avg. value",
        ],
    )
    metric_df.set_index("Datasets", inplace=True)
    metric_df = metric_df.T

    rank_df = metric_df.rank(method="dense", ascending=False).astype(int)
    metric_df = pd.concat([metric_df, rank_df.mean(axis=1)], axis=1).rename(
        columns={0: "Avg. Rank"}
    )
    metric_df["stdDev_mean"] = metric_df[
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
        ]
    ].std(axis=1)
    metric_df["stdDev_rank"] = rank_df[
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
        ]
    ].std(axis=1)
    with open("results/{}_table.tex".format(metric), "w", encoding="utf-8") as f:
        f.write(metric_df.to_latex(index=True))

    # Statistical tests

    # Nemenyi
    cd = utils.compute_CD(
        metric_df["Avg. Rank"].to_list(),
        9,
        alpha=0.05,
    )  # tested on 7 datasets
    utils.graph_ranks(
        metric_df["Avg. Rank"].to_list(),
        model_names_list,
        cd=cd,
        width=10,
        textspace=1.5,
        filename="graphs/Critical_diagram_{}.png".format(metric),
    )

    # Friedman and Kruskal-Wallis H-test
    metric_dict = metric_df.T.to_dict(orient="list")
    NN_test = metric_dict["NN"][0:-2]
    NN_TL_test = metric_dict["NN_TL"][0:-2]
    NN_TLH_test = metric_dict["NN_TLH"][0:-2]
    NN_TLH_sampling_test = metric_dict["NN_TLH_sampling"][0:-2]
    NN_TLH_fifo_test = metric_dict["NN_TLH_fifo"][0:-2]
    NN_TLH_memories_test = metric_dict["NN_TLH_memories"][0:-2]
    NN_TLH_mini_memories_test = metric_dict["NN_TLH_mini_memories"][0:-2]
    NN_TLH_mini_memories_deep_test = metric_dict["NN_TLH_mini_memories_deep"][0:-2]
    BR_HT_test = metric_dict["BR_HT"][0:-2]
    LC_HT_test = metric_dict["LC_HT"][0:-2]
    CC_HT_test = metric_dict["CC_HT"][0:-2]
    BR_random_forest_test = metric_dict["BR_random_forest"][0:-2]
    iSOUPtree_test = metric_dict["iSOUPtree"][0:-2]
    baseline_last_class_test = metric_dict["baseline_last_class"][0:-2]
    baseline_prior_distribution_test = metric_dict["baseline_prior_distribution"][0:-2]
    baseline_mean_test = metric_dict["baseline_mean"][0:-2]
    baseline_oracle_test = metric_dict["baseline_oracle"][0:-2]
    baseline_1_NN_test = metric_dict["baseline_1_NN"][0:-2]
    friedman_results = stats.friedmanchisquare(
        NN_test,
        NN_TL_test,
        NN_TLH_test,
        NN_TLH_sampling_test,
        NN_TLH_fifo_test,
        NN_TLH_memories_test,
        NN_TLH_mini_memories_test,
        NN_TLH_mini_memories_deep_test,
        BR_HT_test,
        LC_HT_test,
        CC_HT_test,
        BR_random_forest_test,
        iSOUPtree_test,
        baseline_last_class_test,
        baseline_prior_distribution_test,
        baseline_mean_test,
        baseline_oracle_test,
        baseline_1_NN_test,
    )
    data = np.array(
        [
            NN_test,
            NN_TL_test,
            NN_TLH_test,
            NN_TLH_sampling_test,
            NN_TLH_fifo_test,
            NN_TLH_memories_test,
            NN_TLH_mini_memories_test,
            NN_TLH_mini_memories_deep_test,
            BR_HT_test,
            LC_HT_test,
            CC_HT_test,
            BR_random_forest_test,
            iSOUPtree_test,
            baseline_last_class_test,
            baseline_prior_distribution_test,
            baseline_mean_test,
            baseline_oracle_test,
            baseline_1_NN_test,
        ]
    )
    nemenyi_results = sp.posthoc_nemenyi(data)
    np.savetxt("results/Nemenyi_{}.txt".format(metric), nemenyi_results)
    output_file = workdir + "results/Friedman_{}.json".format(metric)
    with open(output_file, "w") as outfile:
        json.dump(friedman_results, outfile)


# Pour les métriques continual :
for metric in metrics_continual:
    metric_dict = {}
    for tested_model in models:
        model_perf = []
        sum = 0
        metric_count = 0
        for set_data in datasets:
            with open(
                workdir
                + "results/{}_final_continual_".format(set_data)
                + "{}_".format(tested_model)
                + "{}_".format(scenario)
                + "{}_results.json".format(ordering),
                "r",
                encoding="utf-8",
            ) as json_file:
                continual_file = json.load(json_file)
                model_perf.append(continual_file[metric])
                if not isinstance(continual_file[metric], str):
                    sum += continual_file[metric]
                    metric_count += 1
        if metric_count != 0:
            mean = sum / metric_count
        else:
            mean = 0
        if metric == "average_accuracy":
            coord_consumption[tested_model] = [mean]
            coord_time[tested_model] = [mean]
        model_perf.append(mean)
        metric_dict[tested_model] = model_perf

    metric_df = pd.DataFrame(metric_dict)
    metric_df.insert(
        0,
        "Datasets",
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
            "Avg. value",
        ],
    )
    metric_df.set_index("Datasets", inplace=True)
    metric_df = metric_df.T
    rank_df = metric_df.rank(method="dense", ascending=False).astype(int)
    metric_df = pd.concat([metric_df, rank_df.mean(axis=1)], axis=1).rename(
        columns={0: "Avg. Rank"}
    )
    metric_df["stdDev_mean"] = metric_df[
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
        ]
    ].std(axis=1)
    metric_df["stdDev_rank"] = rank_df[
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
        ]
    ].std(axis=1)
    with open("results/{}_table.tex".format(metric), "w", encoding="utf-8") as f:
        f.write(metric_df.to_latex(index=True))

    # Statistical tests

    # wilco_home

    # Nemenyi
    cd = utils.compute_CD(
        metric_df["Avg. Rank"].to_list(),
        9,
        alpha=0.05,
    )  # tested on 7 datasets
    utils.graph_ranks(
        metric_df["Avg. Rank"].to_list(),
        model_names_list,
        cd=cd,
        width=10,
        textspace=1.5,
        filename="graphs/Critical_diagram_{}.png".format(metric),
    )

    # Friedman
    metric_dict = metric_df.T.to_dict(orient="list")
    NN_test = metric_dict["NN"][0:-2]
    NN_TL_test = metric_dict["NN_TL"][0:-2]
    NN_TLH_test = metric_dict["NN_TLH"][0:-2]
    NN_TLH_sampling_test = metric_dict["NN_TLH_sampling"][0:-2]
    NN_TLH_fifo_test = metric_dict["NN_TLH_fifo"][0:-2]
    NN_TLH_memories_test = metric_dict["NN_TLH_memories"][0:-2]
    NN_TLH_mini_memories_test = metric_dict["NN_TLH_mini_memories"][0:-2]
    NN_TLH_mini_memories_deep_test = metric_dict["NN_TLH_mini_memories_deep"][0:-2]
    BR_HT_test = metric_dict["BR_HT"][0:-2]
    LC_HT_test = metric_dict["LC_HT"][0:-2]
    CC_HT_test = metric_dict["CC_HT"][0:-2]
    BR_random_forest_test = metric_dict["BR_random_forest"][0:-2]
    iSOUPtree_test = metric_dict["iSOUPtree"][0:-2]
    baseline_last_class_test = metric_dict["baseline_last_class"][0:-2]
    baseline_prior_distribution_test = metric_dict["baseline_prior_distribution"][0:-2]
    baseline_mean_test = metric_dict["baseline_mean"][0:-2]
    baseline_oracle_test = metric_dict["baseline_oracle"][0:-2]
    baseline_1_NN_test = metric_dict["baseline_1_NN"][0:-2]
    friedman_results = stats.friedmanchisquare(
        NN_test,
        NN_TL_test,
        NN_TLH_test,
        NN_TLH_sampling_test,
        NN_TLH_fifo_test,
        NN_TLH_memories_test,
        NN_TLH_mini_memories_test,
        NN_TLH_mini_memories_deep_test,
        BR_HT_test,
        LC_HT_test,
        CC_HT_test,
        BR_random_forest_test,
        iSOUPtree_test,
        baseline_last_class_test,
        baseline_prior_distribution_test,
        baseline_mean_test,
        baseline_oracle_test,
        baseline_1_NN_test,
    )
    data = np.array(
        [
            NN_test,
            NN_TL_test,
            NN_TLH_test,
            NN_TLH_sampling_test,
            NN_TLH_fifo_test,
            NN_TLH_memories_test,
            NN_TLH_mini_memories_test,
            NN_TLH_mini_memories_deep_test,
            BR_HT_test,
            LC_HT_test,
            CC_HT_test,
            BR_random_forest_test,
            iSOUPtree_test,
            baseline_last_class_test,
            baseline_prior_distribution_test,
            baseline_mean_test,
            baseline_oracle_test,
            baseline_1_NN_test,
        ]
    )
    nemenyi_results = sp.posthoc_nemenyi(data)
    np.savetxt("results/Nemenyi_{}.txt".format(metric), nemenyi_results)
    output_file = workdir + "results/Friedman_{}.json".format(metric)
    with open(output_file, "w") as outfile:
        json.dump(friedman_results, outfile)


# Pour les métriques de conso :
for metric in metrics_consumption:
    metric_dict = {}
    for tested_model in models:
        model_perf = []
        sum = 0
        metric_count = 0
        for set_data in datasets:
            consumption_df = pd.read_csv(
                workdir
                + "consumption/{}_".format(set_data)
                + "{}_".format(tested_model)
                + "{}_".format(ordering)
                + "{}_consumption.csv".format(scenario)
            )
            model_perf.append(consumption_df.loc[0][metric])
            sum += consumption_df.loc[0][metric]
            metric_count += 1
        mean = sum / metric_count
        if metric == "energy_consumed":
            coord_consumption[tested_model].append(mean)
        if metric == "duration":
            coord_time[tested_model].append(mean)
        model_perf.append(mean)
        metric_dict[tested_model] = model_perf

    metric_df = pd.DataFrame(metric_dict)
    metric_df.insert(
        0,
        "Datasets",
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
            "Avg. value",
        ],
    )
    metric_df.set_index("Datasets", inplace=True)
    metric_df = metric_df.T
    rank_df = metric_df.rank(method="dense", ascending=True).astype(int)
    metric_df = pd.concat([metric_df, rank_df.mean(axis=1)], axis=1).rename(
        columns={0: "Avg. Rank"}
    )
    metric_df["stdDev_mean"] = metric_df[
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
        ]
    ].std(axis=1)
    metric_df["stdDev_rank"] = rank_df[
        [
            "synthetic_monolab",
            "synthetic_bilab",
            "synthetic_rand",
            "Scene",
            "Yeast",
            "Slashdot",
            "Reuters-K500",
            "20NG",
            "Mediamill",
        ]
    ].std(axis=1)
    with open("consumption/{}_table.tex".format(metric), "w", encoding="utf-8") as f:
        f.write(metric_df.to_latex(index=True))

    # Statistical tests

    # Nemenyi
    cd = utils.compute_CD(
        metric_df["Avg. Rank"].to_list(),
        9,
        alpha=0.05,
    )  # tested on 7 datasets
    utils.graph_ranks(
        metric_df["Avg. Rank"].to_list(),
        model_names_list,
        cd=cd,
        width=10,
        textspace=1.5,
        filename="graphs/Critical_diagram_{}.png".format(metric),
    )

    # Friedman
    metric_dict = metric_df.T.to_dict(orient="list")
    NN_test = metric_dict["NN"][0:-2]
    NN_TL_test = metric_dict["NN_TL"][0:-2]
    NN_TLH_test = metric_dict["NN_TLH"][0:-2]
    NN_TLH_sampling_test = metric_dict["NN_TLH_sampling"][0:-2]
    NN_TLH_fifo_test = metric_dict["NN_TLH_fifo"][0:-2]
    NN_TLH_memories_test = metric_dict["NN_TLH_memories"][0:-2]
    NN_TLH_mini_memories_test = metric_dict["NN_TLH_mini_memories"][0:-2]
    NN_TLH_mini_memories_deep_test = metric_dict["NN_TLH_mini_memories_deep"][0:-2]
    BR_HT_test = metric_dict["BR_HT"][0:-2]
    LC_HT_test = metric_dict["LC_HT"][0:-2]
    CC_HT_test = metric_dict["CC_HT"][0:-2]
    BR_random_forest_test = metric_dict["BR_random_forest"][0:-2]
    iSOUPtree_test = metric_dict["iSOUPtree"][0:-2]
    baseline_last_class_test = metric_dict["baseline_last_class"][0:-2]
    baseline_prior_distribution_test = metric_dict["baseline_prior_distribution"][0:-2]
    baseline_mean_test = metric_dict["baseline_mean"][0:-2]
    baseline_oracle_test = metric_dict["baseline_oracle"][0:-2]
    baseline_1_NN_test = metric_dict["baseline_1_NN"][0:-2]
    friedman_results = stats.friedmanchisquare(
        NN_test,
        NN_TL_test,
        NN_TLH_test,
        NN_TLH_sampling_test,
        NN_TLH_fifo_test,
        NN_TLH_memories_test,
        NN_TLH_mini_memories_test,
        NN_TLH_mini_memories_deep_test,
        BR_HT_test,
        LC_HT_test,
        CC_HT_test,
        BR_random_forest_test,
        iSOUPtree_test,
        baseline_last_class_test,
        baseline_prior_distribution_test,
        baseline_mean_test,
        baseline_oracle_test,
        baseline_1_NN_test,
    )
    data = np.array(
        [
            NN_test,
            NN_TL_test,
            NN_TLH_test,
            NN_TLH_sampling_test,
            NN_TLH_fifo_test,
            NN_TLH_memories_test,
            NN_TLH_mini_memories_test,
            NN_TLH_mini_memories_deep_test,
            BR_HT_test,
            LC_HT_test,
            CC_HT_test,
            BR_random_forest_test,
            iSOUPtree_test,
            baseline_last_class_test,
            baseline_prior_distribution_test,
            baseline_mean_test,
            baseline_oracle_test,
            baseline_1_NN_test,
        ]
    )
    nemenyi_results = sp.posthoc_nemenyi(data)
    np.savetxt("results/Nemenyi_{}.txt".format(metric), nemenyi_results)
    output_file = workdir + "results/Friedman_{}.json".format(metric)
    with open(output_file, "w") as outfile:
        json.dump(friedman_results, outfile)

### Plotting the consumption against accuracy :
plt.figure(figsize=(10, 5), dpi=600)
i = 0
texts = []
for k, v in coord_consumption.items():
    plt.scatter(coord_consumption[k][0], coord_consumption[k][1], c=color_list[i])
    texts.append(
        plt.text(
            coord_consumption[k][0],
            coord_consumption[k][1],
            k,
            ha="center",
            va="center",
        )
    )
    i += 1
adjust_text(
    texts,
    expand=(2, 2),
    # force_text=(1, 1),
    # force_static=(1, 1),
    arrowprops=dict(arrowstyle="->", color="red"),
)
plt.ylabel("Energy consumption (kWh)")
plt.xlabel("Average accuracy")
plt.xlim(left=0.5)
plt.ylim(bottom=0)
plt.title("Mean consumption and average accuracy")
plt.tight_layout()
plt.savefig(
    workdir
    + "graphs/0_Final_graph_consumption_{}_".format(ordering)
    + "{}".format(scenario),
    bbox_inches="tight",
)
plt.close()

### Plotting the duration against accuracy :
plt.figure(figsize=(10, 5), dpi=600)
j = 0
texts = []
for k, v in coord_time.items():
    plt.scatter(coord_time[k][0], coord_time[k][1], c=color_list[j])
    texts.append(
        plt.text(
            coord_time[k][0],
            coord_time[k][1],
            model_names[k],
            ha="center",
            va="center",
        )
    )
    j += 1
adjust_text(
    texts,
    expand=(2, 2),
    # force_text=(1, 1),
    # force_static=(1, 1),
    arrowprops=dict(arrowstyle="->", color="red"),
)
plt.ylabel("Experimentation duration (s)")
plt.xlabel("Average accuracy")
plt.xlim(0.5, 0.8)
plt.ylim(bottom=0)
plt.title("Mean duration and average accuracy")
plt.tight_layout()
plt.savefig(
    workdir
    + "graphs/0_Final_graph_duration_{}_".format(ordering)
    + "{}".format(scenario),
    bbox_inches="tight",
)
plt.close()
