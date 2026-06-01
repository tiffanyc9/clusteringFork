# %% imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import os
from sklearn.preprocessing import StandardScaler
import logging

_LABEL_COLUMNS = ['y_true', 'y_live']

_FAR_1D_ANOMALY_CLUSTER = np.array([
    [150, -1, -1],
    [151, -1, -1],
    [152, -1, -1],
    [155, -1, -1],
    [156, -1, -1],
    [157, -1, -1],
    [158, -1, -1],
    [159, -1, -1],
    [151, -1, -1],
    [150, -1, -1],
    [151, -1, -1],
    [152, -1, -1],
    [155, -1, -1],
    [156, -1, -1],
    [157, -1, -1],
    [158, -1, -1],
    [159, -1, -1],
    [151, -1, -1],
])


def _add_rounded_noise(df, feature_column):
    noise = np.round(np.random.normal(0, 1, df.shape[0]), 1) * 0.1
    df[feature_column] += noise


def _append_labelled_rows(df, rows, feature_name):
    extra_df = pd.DataFrame(rows, columns=[feature_name, *_LABEL_COLUMNS])
    return pd.concat([df, extra_df], ignore_index=True)


def _log_label_summary(df):
    num_unlabelled = (df['y_live'] == -1).sum()
    num_labelled = (df['y_live'] != -1).sum()
    labelled_pct = round(num_labelled / df.shape[0] * 100, 2)

    logging.info("Number of unlabelled examples: %s", num_unlabelled)
    logging.info("Number of labelled examples: %s", num_labelled)
    logging.info("Percentage of labelled data: %s%%", labelled_pct)


def _assign_random_live_labels(df, percent_labelled):
    df['y_live'] = -1
    mask = np.random.choice(
        np.arange(len(df)),
        size=int(len(df) * percent_labelled),
        replace=False,
    )
    df.loc[mask, 'y_live'] = df.loc[mask, 'y_true']


def _assign_balanced_live_labels(df, labelled_fraction, rng=None):
    df['y_live'] = -1
    true_classes = df[df['y_true'] >= 0]['y_true'].unique()
    labelled_total = int(labelled_fraction * len(df))
    labelled_per_class = max(1, labelled_total // len(true_classes))

    chooser = np.random.choice if rng is None else rng.choice
    labelled_indices = []

    for cls in true_classes:
        cls_indices = df[df['y_true'] == cls].index
        n_samples = min(labelled_per_class, len(cls_indices))
        labelled_indices.extend(chooser(cls_indices, size=n_samples, replace=False))

    df.loc[labelled_indices, 'y_live'] = df.loc[labelled_indices, 'y_true']


def _make_plot_title(dataset_name, histogram_overlay=False):
    suffix = ' (all data with histogram overlay)' if histogram_overlay else ' (all data)'
    return dataset_name + suffix


def _get_feature_columns(df):
    return [col for col in df.columns if col not in set(_LABEL_COLUMNS)]


def _standardise_features(df, feature_columns, dataset_name):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    logging.debug("Features for dataset '%s' have been standardized.", dataset_name)


def generate_clustering_1d_data(repeat_const=100, percent_labelled=0.03, random_state=None):
    """
    Generates a hand-crafted 1D dataset with repeated clusters, anomalies, and partial labels.

    Parameters:
    - repeat_const (int): Number of times to repeat base data for density.
    - percent_labelled (float): Fraction of data to mark as labelled.
    - random_state (int): Seed for reproducibility.

    Returns:
    - df (pd.DataFrame): DataFrame containing 'X', 'y_true', and 'y_live'.
    """

    # Define base cluster data (labelled)
    data_main = np.array([
        [2.1, 0],
        [2.6, 0],
        [2.4, 0],
        [2.5, 0],
        [2.3, 0],
        [2.1, 0],
        [2.3, 0],
        [2.6, 0],
        [2.6, 0],
        [2.0, 0],
        [2.1, 0],
        [2.0, 0],
        [1.9, 0],
        [2.1, 0],
        [1.8, 0],
        [2.9, 0],

        [56, 1],
        [55, 1],
        [56, 1],
        [58, 1],
        [59, 1],
        [57, 1],
        [56, 1],
        [55, 1],
        [55, 1],
        [55, 1],
        [56.3, 1],
        [55.3, 1],
        [51, 1],
        [56, 1],
        [54.4, 1],
        [57, 1],
        [56, 1],
        [52, 1],
        [53, 1],
        [51, 1],
        [51, 1],
        [50, 1],

        [100, 2],
        [101, 2],
        [102, 2],
        [105, 2],
        [110, 2],
        [108, 2],
        [107, 2],
        [106, 2],
        [111, 2],

        [100, 2],
        [103, 2],
        [101.3, 2],
        [101.8, 2],
        [101.2, 2],
        [109, 2],
        [108, 2],
        [108, 2],
        [111, 2],
        [111, 2],
    ])

    # Anomalies and mislabelled points
    data_anomalies_mislablled = np.array([
        [61, 1],
        [58, 1],
        [8.2, -1],
        [8.3, -1],
        [25, -1],
        [40, -1],
        [80, -1],
        [95, -1],
        [112, 2],
    ])

    # Repeat main data
    data_main_repeated = np.repeat(data_main, repeat_const, axis=0)

    # Combine with anomalies
    data = np.concatenate((data_main_repeated, data_anomalies_mislablled), axis=0)

    # Shuffle rows
    np.random.shuffle(data)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['X', 'y_true'])
    df['y_true'] = df['y_true'].astype(int)

    _add_rounded_noise(df, 'X')
    _assign_random_live_labels(df, percent_labelled)
    df = _append_labelled_rows(df, _FAR_1D_ANOMALY_CLUSTER, 'X')

    return df


def generate_clustering_1d_gauss_anomalies(random_seed=None,
                                            labelled_percent=0.01,
                                            cluster_params=[(0, 1), (50, 3), (100, 8)],   
                                            samples_per_cluster=[10000, 10000, 10000],
                                            include_anomaly_cluster=True):
    """
    Generate 1D synthetic data using Gaussian blobs and inject anomalies.

    Parameters:
    - random_seed (int): For reproducibility.
    - labelled_percent (float): Percentage of points to label.
    - cluster_params (list): (mean, std) tuples for clusters.
    - samples_per_cluster (list): Samples per Gaussian cluster.
    - include_anomaly_cluster (bool): Whether to include a dense anomaly cluster.

    Returns:
    - df (DataFrame): Data with 'X', 'y_true', 'y_live' columns.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate Gaussian clusters
    data_holder = []
    for i, (mu, sig) in enumerate(cluster_params):
        X = np.random.normal(loc=mu, scale=sig, size=samples_per_cluster[i]).reshape(-1, 1)
        y = np.full((samples_per_cluster[i], 1), i)
        data_holder.append(np.hstack([X, y]))

    data_main = np.vstack(data_holder)

    # Inject predefined anomalies/mislabelled points
    anomalies_manual = np.array([
        [61, 1],
        [58, 1],
        [8.2, -1],
        [8.3, -1],
        [25, -1],
        [40, -1],
        [70, -1],
        [80, -1],
        [95, -1],
        [112, 2],
    ])
    data = np.vstack([data_main, anomalies_manual])

    # Shuffle and convert to DataFrame
    np.random.shuffle(data)
    df = pd.DataFrame(data, columns=['X', 'y_true'])
    df['y_true'] = df['y_true'].astype(int)

    _add_rounded_noise(df, 'X')
    _assign_balanced_live_labels(df, labelled_percent)

    # Optionally, add an unknown anomaly cluster far from existing clusters
    if include_anomaly_cluster:
        df = _append_labelled_rows(df, _FAR_1D_ANOMALY_CLUSTER, 'X')

    _log_label_summary(df)
    return df


def generate_clustering_2d_gauss_data(
        n_samples=10000,
        n_components=8,
        num_features=2,
        rand_seed=None,
        same_density=False,
        labelled_fraction=0.01,
        add_anomaly_cluster=True,
        std_dev=None,
    ):
    """
    Generate 2D synthetic dataset using Gaussian blobs with optional anomaly injection.

    Parameters:
    - n_samples (int): Number of main data samples.
    - n_components (int): Number of Gaussian blobs.
    - num_features (int): Dimensionality (default 2D).
    - rand_seed (int): Random seed.
    - same_density (bool): Use identical standard deviation for all clusters.
    - labelled_fraction (float): Fraction of samples to label.
    - add_anomaly_cluster (bool): Whether to inject an anomaly cluster.

    Returns:
    - df (pd.DataFrame): DataFrame with 'f0', ..., 'fN', 'y_true', and 'y_live'.
    """

    np.random.seed(rand_seed)
    
    # Generate the main clusters
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_components,
        n_features=num_features,
        cluster_std=std_dev,
        random_state=rand_seed
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(num_features)])
    df['y_true'] = y_true

    # Randomly label a small fraction of points
    _assign_random_live_labels(df, labelled_fraction)
    _log_label_summary(df)

    # Add anomaly cluster if requested
    if add_anomaly_cluster:
        X_anom, _ = make_blobs(
            n_samples=300,
            centers=[(10, 10), (10, 20), (0, 10)],#[list(range(num_features)), list(range(num_features))],
            n_features=num_features,
            cluster_std=[8.6, 0.2, 10],
            random_state=rand_seed+1
        )
        df_anom = pd.DataFrame(X_anom, columns=[f"f{i}" for i in range(num_features)])
        df_anom['y_true'] = -1
        df_anom['y_live'] = -1
        df = pd.concat([df, df_anom], ignore_index=True)

    return df


def prepare_and_seed_dataset(dataset_name, percent_labelled, k, random_seed, label_column):
    # Read data from the processed folder CSV
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_file_path = os.path.join(project_root, "data", "processed", f"{dataset_name}.csv")
    df = pd.read_csv(csv_file_path)

    # Ensure label column exists
    if label_column not in df.columns:
        raise ValueError(f"The specified label column '{label_column}' was not found in the dataset.")

    # Rename label column to 'y_true'
    df.rename(columns={label_column: 'y_true'}, inplace=True)

    # Determine number of unique classes
    if k is None:
        k = df['y_true'].nunique()

    # Use local random generator for reproducibility
    rng = np.random.default_rng(random_seed)
    logging.debug("Random seed used for sampling: %s", random_seed)
    _assign_balanced_live_labels(df, percent_labelled, rng=rng)

    return df, k


def _load_generated_dataset(dataset_name, random_seed, k, percent_labelled):
    if dataset_name == "1d_simple":
        df = generate_clustering_1d_data(
            repeat_const=100,
            percent_labelled=percent_labelled,
            random_state=random_seed,
        )
        return df, k, _make_plot_title(dataset_name, histogram_overlay=True)

    if dataset_name == "1d_gauss":
        df = generate_clustering_1d_gauss_anomalies(
            random_seed=random_seed,
            labelled_percent=percent_labelled,
            cluster_params=[(0, 1), (50, 3), (100, 8)],
            samples_per_cluster=[10000, 5000, 2500],
            include_anomaly_cluster=True,
        )
        return df, k, _make_plot_title(dataset_name, histogram_overlay=True)

    num_samples = 10000
    same_density = False
    std_dev = 0.6 if same_density else [0.6, 2, 0.2, 0.7, 3, 0.4, 0.6, 0.6][:k]

    df = generate_clustering_2d_gauss_data(
        n_samples=num_samples,
        n_components=k,
        num_features=2,
        rand_seed=random_seed,
        same_density=same_density,
        labelled_fraction=percent_labelled,
        add_anomaly_cluster=True,
        std_dev=std_dev,
    )
    return df, k, _make_plot_title(dataset_name)


def load_dataset(dataset_name, random_seed, k, percent_labelled, standardise):
    """
    Loads a specified dataset, optionally standardizing its features.

    Args:
        dataset_name (str): The name of the dataset to load (e.g., "1d_simple", "2d_gauss").
        random_seed (int): Seed for random number generation to ensure reproducibility.
        k (int): The number of clusters to use or aim for (for semi-supervised settings).
        percent_labelled (float): The percentage of data points to be considered labelled.
        standardise (bool): If True, features will be standardized (mean=0, std=1).

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The loaded (and optionally standardized) DataFrame.
            - num_clusters (int): The number of clusters in the dataset.
            - plot_title (str): A title for plotting.
            - feature_columns (list): A list of column names that are features.
    """
    if dataset_name in {"1d_simple", "1d_gauss", "2d_gauss"}:
        df, num_clusters, plot_title = _load_generated_dataset(
            dataset_name,
            random_seed,
            k,
            percent_labelled,
        )
    else:
        df, num_clusters = prepare_and_seed_dataset(
            dataset_name,
            label_column='class',
            percent_labelled=percent_labelled,
            k=k,
            random_seed=random_seed,
        )
        plot_title = _make_plot_title(dataset_name)

    feature_columns = _get_feature_columns(df)
    if standardise:
        _standardise_features(df, feature_columns, dataset_name)

    return df, num_clusters, plot_title, feature_columns
