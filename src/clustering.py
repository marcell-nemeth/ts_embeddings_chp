import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, single
from scipy.spatial.distance import pdist
from sklearn.cluster import dbscan, k_means
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from tqdm import tqdm


def k_means_(embeddings, n_clusters: int):
    return k_means(embeddings, n_clusters)[1]


def dbscan_(embeddings, n_clusters: int):
    return dbscan(embeddings)[1]


def hierarchical(embeddings, n_clusters: int):
    Z = single(pdist(embeddings))
    return fcluster(Z, n_clusters, criterion="maxclust")


def tsne_dbscan(embeddings, n_clusters: int):
    embeddings = TSNE(n_components=2).fit_transform(embeddings)
    return dbscan(embeddings)[1]


def tsne_hierarchical(embeddings, n_clusters: int):
    embeddings = TSNE(n_components=2).fit_transform(embeddings)
    Z = single(pdist(embeddings))
    return fcluster(Z, n_clusters, criterion="maxclust")


CLUSTERING_METHODS = [
    ("k-means", k_means_),
    ("DBscan", dbscan_),
    ("hierarchical", hierarchical),
    ("tsne-DBscan", tsne_dbscan),
    ("tsne-hierarchical", tsne_hierarchical),
]


def get_tssb_datasets() -> pd.DataFrame:
    import tssb

    prop_filename = Path(tssb.__file__).parent / "datasets" / "properties.txt"
    prop_file = []
    with open(prop_filename, "r") as file:
        for line in file.readlines():
            line = line.split(",")
            ds_name, interpretable, label_cut, resample_rate, labels = (
                line[0],
                bool(line[1]),
                int(line[2]),
                int(line[3]),
                line[4:],
            )
            labels = [int(l.replace("\n", "")) for l in labels]
            prop_file.append((ds_name, label_cut, resample_rate, labels))
    df = pd.DataFrame.from_records(
        prop_file, columns=["dataset", "label_cut", "resample_rate", "labels"]
    )
    df.set_index("dataset", inplace=True)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    method_choices = [name for name, _ in CLUSTERING_METHODS]
    parser.add_argument(
        "--methods",
        nargs="+",
        type=str,
        default="all",
        choices=method_choices,
        help=f"The clustering methods to use. Available options: {', '.join(method_choices)}",
        metavar="METHOD",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main(methods, features):
    datasets = get_tssb_datasets()
    for feature in features:
        feature_path = Path(feature)
        summary_file_path = feature_path / "clustering.csv"
        summary_file_path.unlink(missing_ok=True)
        for method_name, method in methods:

            # Create directory for results
            (feature_path / str(method_name)).mkdir(parents=True, exist_ok=True)

            # Get feature files
            feature_files = list(feature_path.glob("*.csv"))

            # Store calculations and tabular output
            summary = []

            for feature_file in (t := tqdm(feature_files, unit="file")):
                dataset_name = feature_file.name.rstrip(".csv")
                if dataset_name not in datasets.index:
                    continue

                t.set_description(
                    f"Performing {method_name} on {feature_path.name}/{dataset_name}"
                )
                labels = datasets.loc[dataset_name]["labels"]
                n_clusters = len(np.unique(labels))

                clustered_embeddings = [[] for _ in range(n_clusters)]
                with open(feature_file, "r") as f:
                    label2ix = {}
                    for label, embeddings in zip(
                        labels, f.read().strip().split("\n\n")
                    ):
                        if label not in label2ix:
                            label2ix[label] = len(label2ix)
                        ix = label2ix[label]

                        clustered_embeddings[ix].extend(
                            [float(n) for n in embedding.split(";")]
                            for embedding in embeddings.splitlines()[1:]
                        )

                # Turn the embeddings into numpy arrays
                clustered_embeddings = [
                    np.array(embeddings) for embeddings in clustered_embeddings
                ]

                embeddings = np.concatenate(clustered_embeddings)
                clustering_result: np.ndarray = method(embeddings, n_clusters)

                # Calculate adjusted rand index score
                actual_clustering = np.concatenate(
                    [
                        np.array([i] * len(cluster))
                        for i, cluster in enumerate(clustered_embeddings)
                    ]
                )
                ari = adjusted_rand_score(actual_clustering, clustering_result)
                ri = rand_score(actual_clustering, clustering_result)
                summary.append((dataset_name, n_clusters, ari, ri))

                # Transform embeddings into a visualizable form
                embeddings = TSNE(n_components=2).fit_transform(embeddings)

                # Create output
                fig = plt.figure(figsize=(10, 6))
                fig.suptitle(f"{method_name} clustering on {dataset_name}")
                predicted, actual = fig.subplots(nrows=1, ncols=2)
                predicted.grid(True, alpha=0.5)
                actual.grid(True, alpha=0.5)

                predicted.set_title("Predicted clusters")
                _, ix = np.unique(clustering_result, return_index=True)
                for ix in clustering_result[sorted(ix)]:
                    cluster = embeddings[clustering_result == ix]
                    predicted.scatter(cluster[:, 0], cluster[:, 1])

                actual.set_title("Actual clusters")
                left, i = 0, 0
                for cluster in clustered_embeddings:
                    cluster = embeddings[left : left + len(cluster)]
                    actual.scatter(cluster[:, 0], cluster[:, 1], label=f"CLUSTER {i}")
                    left, i = left + len(cluster), i + 1

                fig.legend()
                fig.tight_layout()
                fig.savefig(feature_path / str(method_name) / f"{dataset_name}.png")
                plt.close(fig)

            with open(feature_path / "clustering.csv", "a") as summary_file:
                summary_file.write(f"{method_name}\n")
                summary = pd.DataFrame.from_records(
                    summary, columns=["dataset", "n_classes", "ARI", "RI"]
                )
                summary.to_csv(summary_file, sep=";", index=False)
                summary_file.write("\n")


if __name__ == "__main__":
    args = parse_args()
    main(
        methods=(
            CLUSTERING_METHODS
            if args.methods == "all"
            else [method for method in CLUSTERING_METHODS if method[0] in args.methods]
        ),
        features=args.features,
    )
