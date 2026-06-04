"""
Clustering method wrappers and evaluation runner.

This module is used by the clustering notebooks to apply a shared set of
unsupervised and semi-supervised clustering algorithms to a pandas DataFrame.
Each clustering wrapper expects feature columns, writes predicted cluster
labels back into the DataFrame under a method-specific column, and can
optionally remap arbitrary cluster IDs to known ground-truth labels for easier
comparison.

The bottom half of the file coordinates repeated experiments: it loads dataset
variants, runs the enabled clustering methods, records runtime and dataset size,
and collects the selected evaluation metrics for plotting or inspection.
If you are new to this code, start with `run_metrics_time_clusterings`, then
follow the `clustering_configs` entries passed in from the notebook to see
which wrapper functions are being executed.
"""

import logging
import time
from collections import defaultdict
from itertools import combinations, product

import hdbscan
import numpy as np
import pandas as pd
from clustpy.deep import DEC
from clustering_nassir.cluster import SemiSupervisedClusterer
from copkmeans.cop_kmeans import cop_kmeans
from k_means_constrained import KMeansConstrained
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MeanShift,
    SpectralClustering,
    estimate_bandwidth,
)
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

from utilities.evaluation_metrics import evaluate_prediction_scopes


def remap_clusters_hungarian_with_noise(y_pred, y_true, noise_label=-1):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    all_labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_map = {all_labels[col]: all_labels[row] for row, col in zip(row_ind, col_ind)}

    remapped = np.full_like(y_pred, fill_value=noise_label)
    for idx, label in enumerate(y_pred):
        remapped[idx] = label_map.get(label, noise_label)

    return remapped


def cluster_with_remapping(df, feature_columns, clusterer, target_column='y_true', remap_labels=False):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")

    if not all(col in df.columns for col in feature_columns):
        raise ValueError(f"Some feature columns are missing from the DataFrame: {feature_columns}")

    if remap_labels and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    features = df[feature_columns].to_numpy()
    clusterer.fit(features)
    labels = clusterer.labels_ if hasattr(clusterer, 'labels_') else clusterer.predict(features)

    if remap_labels:
        return remap_clusters_hungarian_with_noise(labels, df[target_column].to_numpy())
    return labels


def _assign_cluster_labels(df, output_column, feature_columns, clusterer, target_column, remap_labels):
    df[output_column] = cluster_with_remapping(
        df,
        feature_columns,
        clusterer,
        target_column,
        remap_labels,
    )
    return df


def kmeans_clustering(df, feature_columns, target_column='y_true',  n_clusters=None,
                      random_state=0, remap_labels=False):
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
    return _assign_cluster_labels(df, 'KMeans', feature_columns, clusterer, target_column, remap_labels)


def meanshift_clustering(df, feature_columns,  n_clusters=None, target_column='y_true', bandwidth=None,
                         remap_labels=False):
    bw = bandwidth or estimate_bandwidth(df[feature_columns].to_numpy(), quantile=0.2, n_samples=500)
    clusterer = MeanShift(bandwidth=bw, bin_seeding=True)
    return _assign_cluster_labels(df, 'MeanShift', feature_columns, clusterer, target_column, remap_labels)


def dbscan_clustering(df, feature_columns,  n_clusters=None, target_column='y_true', eps=0.5, min_samples=5,
                      remap_labels=False):
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    return _assign_cluster_labels(df, 'DBSCAN', feature_columns, clusterer, target_column, remap_labels)


def hdbscan_clustering(df, feature_columns,  n_clusters=None, target_column='y_true', min_cluster_size=5,
                       min_samples=None, remap_labels=False):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    return _assign_cluster_labels(df, 'HDBSCAN', feature_columns, clusterer, target_column, remap_labels)


def agglomerative_clustering(df, feature_columns, target_column='y_true', n_clusters=None,
                             linkage='ward', remap_labels=False):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return _assign_cluster_labels(df, 'Agglomerative', feature_columns, clusterer, target_column, remap_labels)


def gmm_clustering(df, feature_columns,  n_clusters=None, target_column='y_true', remap_labels=False):
    clusterer = GaussianMixture(n_components=n_clusters, random_state=0)
    return _assign_cluster_labels(df, 'GMM', feature_columns, clusterer, target_column, remap_labels)


def spectral_clustering(df, feature_columns, target_column='y_true', n_clusters=None,
                        affinity='nearest_neighbors', remap_labels=False):
    clusterer = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=0)
    return _assign_cluster_labels(df, 'Spectral', feature_columns, clusterer, target_column, remap_labels)


def constrained_kmeans_clustering(df, feature_columns, target_column='y_true',
                                  n_clusters=None, size_min=None, size_max=None,
                                  random_state=0, remap_labels=False):
    n_samples = df[feature_columns].shape[0]
    if size_min is None or size_max is None:
        avg_size = n_samples / n_clusters
        size_min = size_min or max(int(avg_size * 0.5), 1)
        size_max = size_max or int(avg_size * 1.5)

    clusterer = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state,
    )
    return _assign_cluster_labels(df, 'ConstrainedKMeans', feature_columns, clusterer, target_column, remap_labels)


def generate_constraints_from_labels(df, label_column='y_live'):
    must_link = []
    cannot_link = []
    grouped = df[df[label_column] != -1].groupby(label_column)

    for _, group in grouped:
        must_link.extend(combinations(group.index, 2))

    labels = list(grouped.groups.keys())
    for idx, left_label in enumerate(labels):
        for right_label in labels[idx + 1:]:
            cannot_link.extend(product(grouped.groups[left_label], grouped.groups[right_label]))

    return must_link, cannot_link


def copk_means_clustering(df, feature_columns, target_column='y_true', label_column='y_live',
                           n_clusters=None, remap_labels=False):
    must_link, cannot_link = generate_constraints_from_labels(df, label_column=label_column)
    clusters, _ = cop_kmeans(
        dataset=df[feature_columns].to_numpy(),
        k=n_clusters,
        ml=must_link,
        cl=cannot_link,
    )

    if remap_labels and target_column in df.columns:
        df['COPKMeans'] = remap_clusters_hungarian_with_noise(clusters, df[target_column].to_numpy())
    else:
        df['COPKMeans'] = clusters

    return df


def seeded_k_means_clustering(df, feature_columns,  n_clusters=None, target_column='y_true',
                              seeds='y_live', random_state=0,
                              remap_labels=False):
    """
    Perform KMeans clustering with predefined initial centroids
    calculated from the 'y_live' column
    and add a 'SeededKMeans' column to the DataFrame.
    """
    seed_data = df[df[seeds] != -1]

    if not seed_data.empty:
        initial_centroids = seed_data.groupby(seeds)[feature_columns].mean().to_numpy()
        if len(initial_centroids) == n_clusters:
            init = initial_centroids
            n_init = 1
        else:
            logging.warning(
                "Warning: Found %d seed centroids, but n_clusters=%d. Falling back to default init.",
                len(initial_centroids),
                n_clusters,
            )
            init = 'k-means++'
            n_init = 10
    else:
        init = 'k-means++'
        n_init = 10

    clusterer = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)
    return _assign_cluster_labels(df, 'SeededKMeans', feature_columns, clusterer, target_column, remap_labels)


def novel_clustering(df, feature_columns, n_clusters=None, target_column='y_true', seeds='y_live', remap_labels=False):
    """
    Perform clustering using novel clustering method and add a column to the DataFrame.

    Returns:
    - df (pd.DataFrame): DataFrame with predicted cluster labels.
    """
    features_with_seeds = df[feature_columns + [seeds]].to_numpy()
    df['novel_method'] = SemiSupervisedClusterer().fit(features_with_seeds)
    return df


def dec_clustering(df, feature_columns, n_clusters=None,
                   pretrain_epochs=10, clustering_epochs=10,
                   target_column='y_true', remap_labels=True):
    clusterer = DEC(
        n_clusters=n_clusters,
        pretrain_epochs=pretrain_epochs,
        clustering_epochs=clustering_epochs,
    )
    features = df[feature_columns].to_numpy(copy=True)
    clusterer.fit(features)

    if remap_labels and target_column in df.columns:
        df['DEC'] = remap_clusters_hungarian_with_noise(clusterer.labels_, df[target_column].to_numpy())
    else:
        df['DEC'] = clusterer.labels_

    return df


def _has_enough_seeds(df):
    labelled_counts = df[df['y_live'] != -1]['y_live'].value_counts().to_dict()
    return not any(count < 3 for count in labelled_counts.values()), labelled_counts


def _store_last_result(last_results, method_name, df_result):
    last_results[method_name] = df_result[['y_true', 'y_live', method_name]].copy()


def _append_selected_metrics(metrics_accumulator, selected_metrics, method_name, df_result, feature_columns):
    for metric_name, metric_info in selected_metrics.items():
        metric_fn = metric_info["fn"]
        requires_gt = metric_info["requires_gt"]

        try:
            if requires_gt:
                df_filtered = df_result[(df_result['y_true'] != -1) & (df_result[method_name] != -1)]
                score = metric_fn(df_filtered, 'y_true', method_name)
            else:
                df_filtered = df_result[df_result[method_name] != -1]
                score = metric_fn(df_filtered[feature_columns], method_name, feature_columns)

            metrics_accumulator[metric_name][method_name].append(score)
        except Exception as metric_err:
            logging.warning("    Metric %s failed on %s: %s", metric_name, method_name, metric_err)


def _append_full_scope_metrics(metrics_accumulator, selected_metrics, method_name, df_result, feature_columns):
    try:
        evaluation_scores = evaluate_prediction_scopes(
            df=df_result,
            metrics_dict=selected_metrics,
            pred_col=method_name,
            feature_columns=feature_columns,
            true_col="y_true",
            outlier_label=-1,
            retained_suffix=" (retained auxiliary)",
            full_suffix=" (full)",
        )
        for metric_name, score in evaluation_scores.items():
            if (
                metric_name.endswith(" (full)")
                or metric_name in {"Rejected Count", "Rejection Rate"}
            ):
                metrics_accumulator[metric_name][method_name].append(score)
    except Exception as metric_err:
        logging.warning("    Full-dataset evaluation failed on %s: %s", method_name, metric_err)


def _should_skip_method(method_name, dataset_name, clustering_flags, skip_methods):
    if clustering_flags.get(method_name, False) and method_name not in skip_methods:
        return False

    logging.debug(
        "    Skipping clustering method %s for dataset %s due to flag or skip configuration.",
        method_name,
        dataset_name,
    )
    return True


def _get_repeat_count(method_name, dataset_name, num_repeats, methods_to_average):
    if method_name in methods_to_average and dataset_name != 'cover_type':
        return num_repeats
    return 3


def _load_repeat_dataset(load_dataset, dataset_name, random_seed, repeat, k, percent_labelled, standardise):
    return load_dataset(
        dataset_name,
        random_seed + repeat,
        k,
        percent_labelled,
        standardise,
    )


def _run_method_once(config, df, feature_columns, k):
    start = time.time()
    df_result = config['function'](df, feature_columns, n_clusters=k, **config['params'])
    return df_result, time.time() - start


def _record_method_results(
    metrics_accumulator,
    last_results,
    selected_metrics,
    method_name,
    df_result,
    feature_columns,
    elapsed,
):
    metrics_accumulator['runtime (s)'][method_name].append(elapsed)
    metrics_accumulator['Dataset Size'][method_name].append(len(df_result))
    _store_last_result(last_results, method_name, df_result)
    _append_selected_metrics(metrics_accumulator, selected_metrics, method_name, df_result, feature_columns)
    _append_full_scope_metrics(
        metrics_accumulator,
        selected_metrics,
        method_name,
        df_result,
        feature_columns,
    )


def run_metrics_time_clusterings(
    dataset_name,
    clustering_configs,
    clustering_flags,
    skip_clusterings,
    num_repeats=5,
    load_dataset=None,
    random_seed=None,
    k=None,
    percent_labelled=None,
    standardise=False,
    selected_metrics=None,
    num_examples=None,
):
    """
    Run enabled clustering methods for one dataset and collect repeated-run outputs.

    Returns:
        tuple:
            - metrics_by_dataset: nested dict of
              {dataset_name -> {metric_name -> {method_name -> [scores_across_repeats]}}}
            - result_frames_by_method: dict of
              {method_name -> DataFrame[['y_true', 'y_live', method_name]]}
              from the latest successful run, used for plotting/inspection
    """
    logging.debug("\n=== Running clustering algorithms for dataset: %s ===", dataset_name)

    skip_methods = skip_clusterings.get(dataset_name, set())
    metrics_by_dataset = {}
    result_frames_by_method = {}
    metrics_accumulator = defaultdict(lambda: defaultdict(list))

    methods_to_average = {'COPKmeans', 'ConstrainedKMeans', 'SeededKMeans', 'novel_method'}

    for method_name, config in clustering_configs.items():
        if _should_skip_method(method_name, dataset_name, clustering_flags, skip_methods):
            continue

        repeats = _get_repeat_count(method_name, dataset_name, num_repeats, methods_to_average)

        for repeat in range(repeats):
            logging.info("\n--> Running clustering method: %s (Repeat %d/%d)", method_name, repeat + 1, repeats)
            logging.debug("    Parameters: %s", config['params'])

            try:
                df, _, _, feature_columns = _load_repeat_dataset(
                    load_dataset,
                    dataset_name,
                    random_seed,
                    repeat,
                    k,
                    percent_labelled,
                    standardise,
                )
            except Exception as err:
                logging.warning("    Failed to load dataset on repeat %d: %s", repeat + 1, err)
                continue

            enough_seeds, labelled_counts = _has_enough_seeds(df)
            if not enough_seeds:
                logging.warning("    Skipping repeat %d: Not enough seeds per cluster: %s", repeat + 1, labelled_counts)
                continue

            try:
                df_result, elapsed = _run_method_once(config, df, feature_columns, k)
            except Exception as err:
                logging.warning("    ERROR while running %s on repeat %d: %s", method_name, repeat + 1, err)
                continue

            _record_method_results(
                metrics_accumulator,
                result_frames_by_method,
                selected_metrics,
                method_name,
                df_result,
                feature_columns,
                elapsed,
            )

    metrics_by_dataset[dataset_name] = metrics_accumulator
    return metrics_by_dataset, result_frames_by_method
