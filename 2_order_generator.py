import pandas as pd
import argparse
import json
import parameters
import os.path
import random
from sklearn.metrics.pairwise import cosine_similarity

### Setting variables ####################################################################

parser = argparse.ArgumentParser(description="Main")
parser.add_argument(
    "--nb_clusters",
    type=int,
    default=4,
    help="Number of clusters, default to 4",
)
parser.add_argument(
    "--scenario",
    type=str,
    default="task_based",
    help="Other choice : task_free. Default to task_based.",
)
args = parser.parse_args()

nb_clusters = args.nb_clusters
scenario = args.scenario

bench = parameters.benchmark()
datasets = bench.datasets
workdir = bench.workdir

### Functions #############################################################################


def cluster_count(nb_clusters: int, workdir: str):
    """Count the real number of clusters that were created for this dataset.

    Args:
        nb_clusters (int): number of clusters that was used to generate tasks clusters

    Returns:
        real_nb_cluster (int): real number of clusters
    """
    real_nb_cluster = 0

    for i in range(nb_clusters):
        if os.path.isfile(
            workdir + "Eval_set/{}".format(k_data) + "_{}".format(i) + "_eval_set.csv"
        ):
            real_nb_cluster += 1
    return real_nb_cluster


def get_clusters(real_nb_cluster: int, k_data: str, workdir: str, scenario: str):
    """Generates a cluster containing the clusters data.

    Args:
        real_nb_cluster (int): real number of clusters
        k_data (str): name of the dataset

    Returns :
        clusters (dict) : dictionnary containing the data of each clusters.

    """
    clusters = dict()
    for i in range(real_nb_cluster):
        if os.path.isfile(
            workdir
            + "Clusters/{}".format(k_data)
            + "_cluster_{}_".format(i)
            + "{}.csv".format(scenario)
        ):
            clusters[i] = pd.read_csv(
                workdir
                + "Clusters/{}".format(k_data)
                + "_cluster_{}_".format(i)
                + "{}.csv".format(scenario)
            )
    return clusters


def get_centroids(clusters: dict):
    """Generates a dictionnary containing the centroid of each cluster.

    Args:
        clusters (dict): dictionnary containing the data of each cluster.

    Returns:
        centroids (dict): dictionnary containing the centroid of each cluster.
    """
    centroids = dict()
    for k, v in clusters.items():
        centroids[k] = clusters[k].mean().to_numpy().reshape(1, -1)
    return centroids


def get_similarities(centroids: dict):
    """Generates a dictionnary containing the cosine similarities between the cluster centroids.

    Args:
        centroids (dict): dictionnary containing the centroids.

    Returns:
        similarities (dict): dictionnary containing the similarities between centroids.
    """
    similarities = dict()
    for k, v in centroids.items():
        similarities[k] = []
        for key, value in centroids.items():
            if key != k:
                similarities[k].append(cosine_similarity(centroids[k], centroids[key]))
            else:
                similarities[k].append(0)
    return similarities


def get_random_order(real_nb_cluster: int, workdir: str, scenario: str):
    """Generates the csv file containing the cluster order for a random ordering.

    Args:
        real_nb_cluster (int): real number of clusters
    """
    first_order_items = []
    final_order_items = []
    for i in range(real_nb_cluster):
        first_order_items.append(i)
    random.shuffle(first_order_items)
    for i in range(2):
        for j in first_order_items:
            final_order_items.append(j)
    order_idx = dict()
    j = 0
    for i in final_order_items:
        order_idx[j] = i
        j += 1
    print("Random order :")
    print(order_idx)
    with open(
        workdir + "Orders/{}".format(k_data) + "_random_{}.json".format(scenario),
        "w",
        encoding="utf8",
    ) as outfile:
        json.dump(order_idx, outfile)


###########################################################################################
############### Main ######################################################################
###########################################################################################

for k_data, v_data in datasets.items():
    # Count number of clusers
    real_nb_cluster = cluster_count(nb_clusters, workdir)

    # Getting the clusters
    clusters = get_clusters(real_nb_cluster, k_data, workdir, scenario)

    # Getting the centroids of the clusters
    centroids = get_centroids(clusters)

    # Determining the cosine similarities between the centroids of the clusters
    similarities = get_similarities(centroids)

    # Generating the cluster orders
    get_random_order(real_nb_cluster, workdir, scenario)
