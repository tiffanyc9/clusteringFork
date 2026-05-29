from evaluation.clustering_methods import (
    agglomerative_clustering,
    constrained_kmeans_clustering,
    copk_means_clustering,
    dbscan_clustering,
    dec_clustering,
    gmm_clustering,
    hdbscan_clustering,
    kmeans_clustering,
    meanshift_clustering,
    novel_clustering,
    seeded_k_means_clustering,
    spectral_clustering,
)
from utilities.evaluation_metrics import (
    compute_accuracy,
    compute_ari,
    compute_completeness,
    compute_fmi,
    compute_homogeneity,
    compute_nmi,
    compute_purity,
    compute_v_measure,
)


def _dataset(name, percent_labelled, k, plot_figure=False, standardise=False, random_seed=None):
    return {
        "name": name,
        "percent_labelled": percent_labelled,
        "k": k,
        "plot_figure": plot_figure,
        "standardise": standardise,
        "random_seed": random_seed,
    }


def _method(function, **params):
    return {
        "function": function,
        "params": params,
    }


# Datasets are pre-processed so the final column is the integer-encoded class label
# and all preceding columns are named features.
dataset_dict = {
    # 0: _dataset("1d_simple", 0.03, 3),
    1: _dataset("1d_gauss", 0.002, 3),
    2: _dataset("2d_gauss", 0.01, 8, plot_figure=True, random_seed=6772),  # 4549 6628 743 8858 6772
    3: _dataset("iris", 0.2, 3),  # 8338 3480 9093
    4: _dataset("wine", 0.3, 3),  # 3169 9942
    5: _dataset("breast_cancer", 0.07, 2),  # 1451
    6: _dataset("seeds", 0.2, 3),  # 8993
    7: _dataset("glass", 0.3, 6),  # 1986
    8: _dataset("ionosphere_UMAP10", 0.1, 2),  # 4574
    9: _dataset("yeast", 0.05, 4),  # good example for failure analysis
    10: _dataset("banknote", 0.02, 2),  # 21, appears more than 2 clusters
    11: _dataset("pendigits", 0.025, 10),  # 769
    12: _dataset("land_mines", 0.3, 5),
    13: _dataset("MNIST_UMAP10", 0.10, 10),  # 4470
    14: _dataset("6NewsgroupsUMAP10", 0.02, 6),
    15: _dataset("shuttle", 0.002, 3),  # highly imbalanced, one class dominates
    16: _dataset("cover_type", 2e-4, 7),
}


clustering_flags = {
    "KMeans": True,
    "MeanShift": False,
    "DBSCAN": True,
    "HDBSCAN": False,
    "Agglomerative": True,
    "GMM": True,
    "Spectral": False,
    "ConstrainedKMeans": True,
    "COPKMeans": True,
    "SeededKMeans": True,
    "novel_method": True,
    "DEC": True,
}


_REMAP_TO_TRUE = {
    "target_column": "y_true",
    "remap_labels": True,
}

clustering_configs = {
    "KMeans": _method(kmeans_clustering, **_REMAP_TO_TRUE),
    "MeanShift": _method(meanshift_clustering, **_REMAP_TO_TRUE),
    "DBSCAN": _method(dbscan_clustering, **_REMAP_TO_TRUE),
    "HDBSCAN": _method(
        hdbscan_clustering,
        **_REMAP_TO_TRUE,
        min_cluster_size=5,
        min_samples=None,
    ),
    "Agglomerative": _method(agglomerative_clustering, **_REMAP_TO_TRUE),
    "GMM": _method(gmm_clustering, **_REMAP_TO_TRUE),
    "Spectral": _method(spectral_clustering, **_REMAP_TO_TRUE),
    "ConstrainedKMeans": _method(
        constrained_kmeans_clustering,
        **_REMAP_TO_TRUE,
        size_min=15,
        size_max=None,  # set dynamically to df.shape[0] in the method
    ),
    "COPKMeans": _method(
        copk_means_clustering,
        **_REMAP_TO_TRUE,
        label_column="y_live",
    ),
    "SeededKMeans": _method(
        seeded_k_means_clustering,
        **_REMAP_TO_TRUE,
        seeds="y_live",
    ),
    "novel_method": _method(
        novel_clustering,
        target_column="y_true",
        seeds="y_live",
        remap_labels=False,
    ),
    "DEC": _method(
        dec_clustering,
        **_REMAP_TO_TRUE,
        pretrain_epochs=10,   # default 100
        clustering_epochs=10,  # default 150
    ),
}


skip_clustering = {
    "shuttle": {
        "MeanShift",
        "Agglomerative",
        "Spectral",
        "COPKMeans",
    },
    "cover_type": {
        "MeanShift",
        "Agglomerative",
        "Spectral",
        "COPKMeans",
        "DBSCAN",
        "ConstrainedKMeans",
    },
}


selected_metrics = {
    "Accuracy": {"fn": compute_accuracy, "requires_gt": True},
    "Purity": {"fn": compute_purity, "requires_gt": True},
    "Homogeneity": {"fn": compute_homogeneity, "requires_gt": True},
    "Completeness": {"fn": compute_completeness, "requires_gt": True},
    "V-Measure": {"fn": compute_v_measure, "requires_gt": True},
    "NMI": {"fn": compute_nmi, "requires_gt": True},
    "ARI": {"fn": compute_ari, "requires_gt": True},
    "FMI": {"fn": compute_fmi, "requires_gt": True},
}
