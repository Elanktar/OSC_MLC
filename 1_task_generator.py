import argparse
import pandas as pd
import numpy as np
import parameters
import json
import re
import os.path
from coclust import clustering
from skmultilearn.dataset import load_from_arff
from sklearn.metrics.pairwise import cosine_similarity

### Setting variables ########################################################

parser = argparse.ArgumentParser(description="Task generator")
parser.add_argument(
    "--nb_cluster",
    type=int,
    default=4,
    help="Number of tasks you would like to create, default at 4",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Seed of the k-means model you want to use, default at 1.",
)
args = parser.parse_args()

seed = args.seed
nb_cluster = args.nb_cluster

bench = parameters.benchmark()
datasets = bench.datasets
workdir = bench.workdir

### Functions ###################################################################


def null_processing(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    full_data: pd.DataFrame,
    label_count: int,
):
    """Check and delete the null label and/or feature vector. It also delete null columns.

    Args:
        df_x (pd.DataFrame): pandas dataframe of the feature vectors
        df_y (pd.DataFrame): pandas dataframe of the label vectors
        full_data (pd.DataFrame): pandas dataframe of the full dataset (x and y vectors)
        label_count (int): number of labels in the dataset

    Returns:
        label_count(int) : updated number of labels in the dataset
    """
    null_vectors = []
    null_label_column = []
    null_feature_column = []

    for column in df_y:
        if sum(df_y[column]) == 0:
            df_y.drop(column, axis=1, inplace=True)
            full_data.drop(column, axis=1, inplace=True)
            null_label_column.append(int(re.search(r"\d+", column).group()))
            label_count = label_count - 1

    for column in df_x:
        if sum(df_x[column]) == 0:
            df_x.drop(column, axis=1, inplace=True)
            full_data.drop(column, axis=1, inplace=True)
            null_feature_column.append(int(column))

    for n in null_feature_column:
        for col in df_x:
            idx_col = int(col)
            if idx_col > n:
                df_x.rename(
                    columns={"{}".format(idx_col): "{}".format(idx_col - 1)},
                    inplace=True,
                )
                full_data.rename(
                    columns={"{}".format(idx_col): "{}".format(idx_col - 1)},
                    inplace=True,
                )

    i = 0
    for n in null_label_column:
        for col in df_y:
            idx_col = int(re.search(r"\d+", col).group())
            if idx_col > n - i:
                df_y.rename(
                    columns={"L{}".format(idx_col): "L{}".format(idx_col - 1)},
                    inplace=True,
                )
                full_data.rename(
                    columns={"L{}".format(idx_col): "L{}".format(idx_col - 1)},
                    inplace=True,
                )
        i += 1

    for index, row in df_y.iterrows():
        null_count = 0
        for label in row.items():
            null_count += label[1]
        if null_count == 0:
            print(row)
            print(index)
            null_vectors.append(index)

    if null_vectors:
        df_y.drop(null_vectors, inplace=True)
        data.drop(null_vectors, inplace=True)

    print("Null label and/or feature vector indexes :")
    print(null_vectors)

    print("Null feature columns :")
    print(null_feature_column)

    print("Null label columns :")
    print(null_label_column)

    return label_count


def cluster_processing(
    clusters: dict,
    nb_cluster: int,
    labels: list,
    data: pd.DataFrame,
    df_y: pd.DataFrame,
):
    """Generates the cluster labels occurences matrix, and fill the clusters dict with cluster data.

    Args:
        clusters (dict): dictionnary which will contain the cluster data
        nb_cluster (int): number of clusters
        labels (list): list containing each sample cluster label
        data (pd.DataFrame): dataframe containing the full processed dataset
        df_y (pd.DataFrame): dataframe containing the processed label vectors

    Returns:
        matrix (pd.DataFrame): matrix containing the cluster labels occurences.
    """
    matrix = pd.DataFrame()
    for i in range(nb_cluster):
        bool_vector = []
        for j in labels:
            if j == i:
                bool_vector.append(True)
            else:
                bool_vector.append(False)
        filtered_cluster = data.loc[bool_vector]
        clusters[i] = filtered_cluster
        filtered_label = df_y.loc[bool_vector]
        matrix = pd.concat([matrix, filtered_label.sum()], axis=1)
        matrix.rename(columns={0: "Cluster {}".format(i)}, inplace=True)
    return matrix


def label_cluster_attribution(matrix: pd.DataFrame):
    """Fills a dict with the labels by clusters.

    Args:
        matrix (pd.DataFrame): matrix containing the cluster labels occurences.

    Returns:
        cluster_tasks (dict): dict containing the labels by clusters.
    """
    cluster_tasks = dict()

    for index, row in matrix.iterrows():
        for task, occurence in row.items():
            if occurence != 0:
                if task not in cluster_tasks.keys():
                    cluster_tasks[task] = [index]
                else:
                    cluster_tasks[task].append(index)
    return cluster_tasks


def mono_lab_cluster_fusing(
    nb_cluster: int,
    cluster_tasks: dict,
    clusters: dict,
    new_cluster_tasks: dict,
    new_clusters: dict,
):
    """Fuse the mono-label clusters with the closest cluster by centroids cosine similarity.

    Args:
        nb_cluster (int): number of clusters
        cluster_tasks (dict): dictionnary containing the labels by task
        clusters (dict): dictionnary containing the data of each cluster
        new_cluster_tasks (dict): dictionnary that will be filled with the labels by final clusters
        new_clusters (dict): dictionnary that will be filled with the data of each final cluster
    """
    used_cluster = []
    for i in range(nb_cluster):
        if len(cluster_tasks["Cluster {}".format(i)]) == 1:
            candidates_centroids = dict()
            max_sim_index = 0
            max_sim = 0
            mono_centroid = clusters[i].mean().to_numpy().reshape(1, -1)
            for j in range(nb_cluster):
                if j != i and j not in used_cluster:
                    candidates_centroids[j] = (
                        clusters[j].mean().to_numpy().reshape(1, -1)
                    )
            for key, value in candidates_centroids.items():
                sim_candidate = cosine_similarity(value, mono_centroid)
                if max_sim < sim_candidate:
                    max_sim = sim_candidate
                    max_sim_index = key
            new_cluster_tasks["Cluster {}".format(max_sim_index)] = (
                cluster_tasks["Cluster {}".format(max_sim_index)]
                + cluster_tasks["Cluster {}".format(i)]
            )
            new_clusters[max_sim_index] = pd.concat(
                [clusters[max_sim_index], clusters[i]]
            )
            used_cluster.append(i)
            print(
                "Cluster {} and cluster ".format(i)
                + "{} are fused.".format(max_sim_index)
            )
        else:
            new_cluster_tasks["Cluster {}".format(i)] = cluster_tasks[
                "Cluster {}".format(i)
            ]
            new_clusters[i] = clusters[i]


def cluster_labels_diversity(label_count: int, new_cluster_tasks: dict):
    """Check if the cluster label signatures are different with cosine similarity.

    Args:
        label_count (int): number of labels
        new_cluster_tasks (dict): dictionnary containing the labels by cluster.
    """
    cluster_label_signatures = dict()
    for key, value in new_cluster_tasks.items():
        label_signature = np.zeros((label_count))
        for j in value:
            label = int(re.search(r"\d+", j).group())
            label_signature[label] = 1
        cluster_label_signatures[key] = label_signature

    for key, value in cluster_label_signatures.items():
        for kk, vv in cluster_label_signatures.items():
            if not np.array_equal(value, vv):
                cos_similarity = cosine_similarity(
                    value.reshape(1, -1), vv.reshape(1, -1)
                )
                if cos_similarity > 0.99:
                    print(
                        "Same label signatures between the following clusters {}".format(
                            key
                        )
                        + " and {}".format(kk)
                    )


def task_based_scenario_files(new_clusters: dict, k: str, workdir: str, seed: int):
    """Generates the experiences and evaluation sets csv files, as well as a length json file containing the lengths of all the csv files for this dataset and scenario.

    Args:
        new_clusters (dict): dictionnary containing the data of each clusters
        k (str): dataset name
        workdir (str): working directory
        seed (int): seed for randomness
    """
    stream_length = dict()
    cluster_idx = 0
    for key, value in new_clusters.items():
        cluster = value.sample(frac=1, random_state=seed)
        cluster.to_csv(
            workdir
            + "Clusters/{}".format(k)
            + "_cluster_{}_task_based.csv".format(cluster_idx),
            index=False,
        )
        exp_1, exp_2, eval_set = np.array_split(
            cluster, [int(0.35 * len(cluster)), int(0.70 * len(cluster))]
        )
        stream_length["cluster_{}_exp_1".format(cluster_idx)] = exp_1.shape[0]
        stream_length["cluster_{}_exp_2".format(cluster_idx)] = exp_2.shape[0]
        stream_length["cluster_{}_eval_set".format(cluster_idx)] = eval_set.shape[0]
        exp_1.to_csv(
            workdir + "Experiences/{}".format(k) + "_{}_exp_1.csv".format(cluster_idx),
            index=False,
        )
        exp_2.to_csv(
            workdir + "Experiences/{}".format(k) + "_{}_exp_2.csv".format(cluster_idx),
            index=False,
        )
        eval_set.to_csv(
            workdir + "Eval_set/{}".format(k) + "_{}_eval_set.csv".format(cluster_idx),
            index=False,
        )
        temp = new_cluster_tasks.pop("Cluster {}".format(key))
        new_cluster_tasks["Cluster {}".format(cluster_idx)] = temp
        cluster_idx += 1
    output_file = workdir + "Labels/{}_".format(k) + "cluster_labels_task_based.json"
    with open(output_file, "w") as outfile:
        json.dump(new_cluster_tasks, outfile)
    output_file = workdir + "Length/{}_".format(k) + "stream_length_task_based.json"
    with open(output_file, "w") as outfile:
        json.dump(stream_length, outfile)


###########################################################################################
############### Main ######################################################################
###########################################################################################

for k, v in datasets.items():
    # Importing the dataset
    print("Importing and preprocessing the dataset...")

    assert isinstance(
        datasets[k], list
    ), "Label number not found in the datasets dictionnary from parameters file."

    assert os.path.isfile(
        workdir + "datasets/{}.arff".format(k)
    ), "{} dataset not found in datasets directory".format(k)

    X, y = load_from_arff(
        workdir + "datasets/{}.arff".format(k),
        label_count=v[0],
        label_location="start",
        load_sparse=True,
        return_attribute_definitions=False,
    )

    # Creating a label vector pandas dataframe and dataset pandas dataframe
    data_y = y.toarray()
    df_y = pd.DataFrame(data_y)
    for i in df_y.columns.values:
        df_y.rename(columns={i: "L{}".format(i)}, inplace=True)

    data_x = X.toarray()
    df_x = pd.DataFrame(data_x)
    data = pd.concat([df_x, df_y], axis=1)

    # Making sure there is no null label vectors or null label column :
    label_count = null_processing(df_x, df_y, data, v[0])

    # Converting label vectors from pandas dataframe to numpy array :
    data_y = df_y.to_numpy()

    # Clustering :
    print("Clustering...")
    model = clustering.spherical_kmeans.SphericalKmeans(
        n_clusters=nb_cluster,
        random_state=seed,
        weighting=True,
        n_init=10,
    )
    model.fit(data_y)
    labels = model.labels_

    # Cluster processing
    print("Cluster processing...")

    clusters = dict()
    matrix = cluster_processing(clusters, nb_cluster, labels, data, df_y)
    print("Cluster's label's occurences matrix :")
    print(matrix)

    # Attributing labels to each cluster :
    cluster_tasks = label_cluster_attribution(matrix)

    # Fusing mono-label tasks :
    new_cluster_tasks = dict()
    new_clusters = dict()
    mono_lab_cluster_fusing(
        nb_cluster, cluster_tasks, clusters, new_cluster_tasks, new_clusters
    )
    print("Tasks of each cluster :")
    print(new_cluster_tasks)

    # Label signature diversity verification :
    cluster_labels_diversity(label_count, new_cluster_tasks)

    # Creating the experiences and evaluation sets if task-based scenario :
    task_based_scenario_files(new_clusters, k, workdir, seed)
