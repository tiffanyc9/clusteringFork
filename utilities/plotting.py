import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import umap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


_FIGSIZE = (12, 8)
_NOISE_LABEL = -1
_UMAP_RANDOM_STATE = 42

_CUD_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

_BASE_LABEL_COLORS = {
    -1: "red",
    0: "green",
    1: "blue",
    2: "black",
    3: "orange",
    4: "purple",
    5: "brown",
    6: "pink",
    7: "cyan",
    8: "darkblue",
    9: "violet",
    10: "magenta",
}

_EMBEDDING_LABEL_COLUMNS = ["y_true", "y_live", "KMeans", "novel_method"]

_NEWSGROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2",
    "#c7c7c7", "#dbdb8d", "#9edae5", "#393b79", "#637939",
]

_MNIST_COLORS = [
    "green", "blue", "black",
    "orange", "purple", "brown",
    "pink", "cyan", "darkblue",
    "violet", "magenta", "black",
]


def _resolve_colors(labels, colors=None):
    color_map = dict(_BASE_LABEL_COLORS)
    if colors is not None:
        color_map.update(colors)

    unique_labels = sorted(pd.unique(labels))
    missing_labels = [label for label in unique_labels if label not in color_map]

    if missing_labels:
        if len(missing_labels) <= len(_CUD_PALETTE):
            palette = _CUD_PALETTE[: len(missing_labels)]
        else:
            palette = sns.color_palette("hls", n_colors=len(missing_labels))

        for label, color in zip(missing_labels, palette):
            color_map[label] = color

    return color_map


def _project_features(df, feature_columns):
    if not feature_columns:
        raise ValueError("Error: feature_columns must contain at least 1 column")

    plot_df = df.copy()
    plot_columns = list(feature_columns)

    if len(plot_columns) > 2:
        embedding = umap.UMAP(n_components=2, random_state=_UMAP_RANDOM_STATE)
        reduced = embedding.fit_transform(plot_df[plot_columns])
        plot_df["UMAP_1"] = reduced[:, 0]
        plot_df["UMAP_2"] = reduced[:, 1]
        plot_columns = ["UMAP_1", "UMAP_2"]

    return plot_df, plot_columns


def _style_axis(ax, title, legend_label):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=14)

    if title is not None:
        ax.set_title(title, fontsize=18)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            title=legend_label,
            title_fontsize=16,
            fontsize=14,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            markerscale=2.0,
            frameon=False,
        )


def _format_email_body(text, words_per_line=10):
    words = str(text).split()
    lines = [" ".join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
    return "<br>".join(lines)


def _merge_visualisation_results(df_vis, df_one_result, label_columns):
    for label_column in label_columns:
        if label_column in {"y_true", "y_live"}:
            continue
        if label_column not in df_one_result:
            raise KeyError(f"Column '{label_column}' not found in df_one_result")

        result_df = df_one_result[label_column]
        if "y_true" not in df_vis:
            df_vis["y_true"] = result_df["y_true"]
        if "y_live" not in df_vis:
            df_vis["y_live"] = result_df["y_live"]
        df_vis[label_column] = result_df[label_column]

    return df_vis


def _ordered_plotly_labels(df_vis, label_columns):
    all_labels = set()
    for label_column in label_columns:
        if label_column in df_vis:
            all_labels.update(df_vis[label_column].astype(str).unique())

    non_noise_labels = sorted((label for label in all_labels if label != "-1"), key=int)
    return (["-1"] if "-1" in all_labels else []) + non_noise_labels


def _build_plotly_color_map(ordered_labels, colors):
    return {
        label: ("rgb(255,0,0)" if label == "-1" else colors[index % len(colors)])
        for index, label in enumerate(ordered_labels)
    }


def _style_plotly_embedding(fig, label_column):
    for trace in fig.data:
        trace.marker.line.color = "white"
        trace.marker.line.width = 1.5

    fig.update_layout(
        legend_title_text=label_column,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="",
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            ticks="outside",
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            ticks="outside",
        ),
    )


def plot_embedding_label_views(
    dataset_name,
    df_one_result,
    project_root,
    plot_save_path=None,
    save_plots=False,
):
    if dataset_name == "6NewsgroupsUMAP10":
        csv_file_path = os.path.join(project_root, "data", "processed", "6NewsgroupsUMAP2_embeddings.csv")
        df_vis = pd.read_csv(csv_file_path)
        df_vis["email_body_formatted"] = df_vis["email_body"].apply(_format_email_body)
        hover_columns = ["category", "top_keywords", "email_body_formatted"]
        colors = _NEWSGROUP_COLORS
        subset_colors = True
        save_prefix = None
    elif dataset_name == "MNIST_UMAP10":
        csv_file_path = os.path.join(project_root, "data", "processed", "MNIST_UMAP2_with_images.csv")
        df_vis = pd.read_csv(csv_file_path)
        hover_columns = []
        colors = _MNIST_COLORS
        subset_colors = False
        save_prefix = "mnist_umap"
    else:
        return

    df_vis = _merge_visualisation_results(df_vis, df_one_result, _EMBEDDING_LABEL_COLUMNS)
    df_vis["index"] = df_vis.index.astype(str)
    ordered_labels = _ordered_plotly_labels(df_vis, _EMBEDDING_LABEL_COLUMNS)
    global_color_map = _build_plotly_color_map(ordered_labels, colors)

    if save_plots and plot_save_path is not None:
        os.makedirs(plot_save_path, exist_ok=True)

    for label_column in _EMBEDDING_LABEL_COLUMNS:
        if label_column not in df_vis:
            continue

        df_vis[label_column] = df_vis[label_column].astype(str)
        if subset_colors:
            ordered_categories = [label for label in ordered_labels if label in df_vis[label_column].values]
            color_map = {label: global_color_map[label] for label in ordered_categories}
        else:
            ordered_categories = ordered_labels
            color_map = global_color_map

        hover_data = [
            column
            for column in ["index", "y_true", "y_live", "KMeans", "novel_method", *hover_columns]
            if column != label_column and column in df_vis.columns
        ]

        fig = px.scatter(
            df_vis,
            x="UMAP_1",
            y="UMAP_2",
            color=label_column,
            color_discrete_map=color_map,
            category_orders={label_column: ordered_categories},
            hover_name=None,
            hover_data=hover_data,
            title=f"UMAP projection colored by {label_column}",
            width=1400,
            height=900,
        )

        _style_plotly_embedding(fig, label_column)
        fig.show()

        if save_prefix and save_plots and plot_save_path is not None:
            save_file = os.path.join(plot_save_path, f"{save_prefix}_{label_column}.png")
            fig.write_image(save_file, scale=2)


def plot_clusters(
    df,
    feature_columns,
    label_column,
    title=None,
    x_axis_label="",
    y_axis_label=None,
    legend_label="Cluster labels",
    colors=None,
    show_seeds_only=False,
):
    """
    Generic cluster plotting function for 1D, 2D, or multi-dimensional data.

    Parameters:
    - df (pd.DataFrame): The DataFrame with features and label column.
    - feature_columns (list): List of feature column names.
    - label_column (str): Column name containing cluster labels to plot.
    - title (str): Plot title (optional).
    - colors (dict): Optional custom colour palette (label: color).
    - show_seeds_only (bool): Whether to show only seed points (exclude noise).
    """

    plot_df, plot_columns = _project_features(df, feature_columns)
    if show_seeds_only:
        plot_df = plot_df[plot_df[label_column] != _NOISE_LABEL].copy()

    palette = _resolve_colors(df[label_column], colors)
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    if len(plot_columns) == 1:
        xcol = plot_columns[0]

        if not show_seeds_only:
            sns.histplot(plot_df, x=xcol, bins=1000, color="lightgrey", ax=ax)
            baseline = ax.get_ylim()[1] * 0.03
            y_values = np.full(plot_df.shape[0], baseline, dtype=float)
        else:
            y_values = np.zeros(plot_df.shape[0], dtype=float)

        if not plot_df.empty:
            sns.scatterplot(
                data=plot_df,
                x=xcol,
                y=y_values,
                hue=label_column,
                palette=palette,
                legend="full",
                edgecolor="none",
                ax=ax,
            )

        ax.set_xlabel("" if x_axis_label is None else x_axis_label, fontsize=16)
        if y_axis_label is not None:
            ax.set_ylabel(y_axis_label, fontsize=16)
        elif show_seeds_only:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            ax.set_ylabel(ax.get_ylabel(), fontsize=16)

    elif len(plot_columns) == 2:
        if not plot_df.empty:
            sns.scatterplot(
                data=plot_df,
                x=plot_columns[0],
                y=plot_columns[1],
                hue=label_column,
                palette=palette,
                legend="full",
                edgecolor="white",
                ax=ax,
            )

        ax.set_xlabel("" if x_axis_label is None else x_axis_label, fontsize=16)
        ax.set_ylabel("" if y_axis_label is None else y_axis_label, fontsize=16)

    else:
        raise ValueError("Error: feature_columns must contain at least 1 column")

    _style_axis(ax, title, legend_label)
    fig.tight_layout()
    plt.show()
    return fig


def plot_enabled_clusterings(
    df,
    clustering_flags,
    feature_columns,
    plot_save_path=None,
    dataset_name=None,
    save_plots=False,
):
    for name, enabled in clustering_flags.items():
        if not enabled:
            continue

        fig = plot_clusters(
            df,
            feature_columns,
            label_column=name,
            title=f"{name} Clustering",
        )

        if save_plots and plot_save_path is not None:
            filename = f"{name}_{dataset_name}.png"
            filepath = os.path.join(plot_save_path, filename)
            fig.savefig(filepath)
            print(f"Saved plot to {filepath}")


def plot_confusion_matrices_for_clustering(
    df,
    true_label_col,
    clustering_flags,
    title_prefix="Confusion Matrix",
    exclude_noise=False,
):
    """
    Plots confusion matrices for all enabled clustering methods in `clustering_flags`.

    Parameters:
    - df (pd.DataFrame): DataFrame with true and predicted labels.
    - true_label_col (str): Name of the column containing ground-truth labels.
    - clustering_flags (dict): Dictionary of clustering method names with booleans indicating if they are enabled.
    - title_prefix (str): Prefix for plot titles.
    - exclude_noise (bool): Whether to exclude the noise label (-1) from the confusion matrix.
    """

    true_labels = df[true_label_col].values
    enabled_methods = [name for name, enabled in clustering_flags.items() if enabled]

    for method in enabled_methods:
        predicted_labels = df[method].values
        print(f"Unique predicted labels for {method}: {np.unique(predicted_labels)}")

        labels = sorted(np.unique(np.concatenate((true_labels, predicted_labels))))
        if exclude_noise and _NOISE_LABEL in labels:
            labels.remove(_NOISE_LABEL)

        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

        print(f"\nConfusion matrix for {method}:")
        print(pd.DataFrame(cm, index=labels, columns=labels))

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
        ax.set_title(f"{title_prefix} - {method}")
        fig.tight_layout()
        plt.show()
