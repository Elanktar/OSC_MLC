class benchmark:
    def __init__(self):
        self.workdir = ""
        self.datasets = {
            "synthetic_monolab": [4, 4],
            "synthetic_bilab": [4, 4],
            "synthetic_rand": [4, 4],
            "Scene": [6, 294],
            "Yeast": [14, 103],
            "Slashdot": [20, 1079],
            "Reuters-K500": [99, 500],
            "20NG": [20, 1006],
            "Mediamill": [101, 120],
        }
        self.models = [
            # "NN",
            # "NN_TL",
            # "NN_TLH",
            # "NN_TLH_sampling",
            # "NN_TLH_fifo",
            # "NN_TLH_memories",
            # "NN_TLH_mini_memories",
            # "NN_TLH_mini_memories_deep",
            # "BR_HT",
            # "LC_HT",
            # "CC_HT",
            # "BR_random_forest",
            # "iSOUPtree",
            "baseline_last_class",
            "baseline_prior_distribution",
            "baseline_mean",
            "baseline_oracle",
            "baseline_1_NN",
        ]
