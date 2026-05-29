# %%
# ---------------------------- Imports and Setup ----------------------------
import sys
from pathlib import Path
import random
import pickle
import pandas as pd
from IPython.display import display
pd.set_option('display.max_rows', 200)

# Project Root Resolution and add to sys.path
CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT_PATH = CURRENT_DIR.parent
sys.path.insert(0, str(ROOT_PATH))  

# Internal Module Imports 
from evaluation.evaluation_configs import dataset_dict, clustering_configs, clustering_flags, selected_metrics, skip_clustering
from evaluation.clustering_methods import run_metrics_time_clusterings
from utilities.plotting import (
    plot_clusters,
    plot_enabled_clusterings,
    plot_confusion_matrices_for_clustering,
    plot_embedding_label_views,
)
from utilities.cluster_utilities import (
    create_full_comparison_table,
    create_metric_tables,
    create_proposed_full_dataset_table,
    display_table,
    median_metrics_dataframe,
    metrics_to_dataframe,
    save_table_csv,
    save_metric_tables_as_latex,
)
from utilities.generate_load_data import load_dataset
from utilities.run_utilities import finish_profiler, setup_logging, start_profiler

# %%
# -------------------------- Experiment Configuration ------------------------
class Config:
    """Centralised configuration settings."""
    PROFILE_CODE = False # Enable to profile code execution time
    IS_TESTING = False   # for producing more verbose output during development
    PLOT_FIGURES = False # Enable to plot figures for each dataset
    SAVE_RESULTS = False # Save all output tables
    SAVE_REBUTTAL_RESULTS = False # Save only rebuttal tables
    SAVE_PLOTS = False   # Save plots to disk
    RESULTS_FOLDER = Path("results") # Folder to save results
    PLOT_SAVE_PATH = Path.home() / "Google Drive/docs/A_computational_theory_of_clustering/figures"
    TABLE_SAVE_PATH = Path.home() / "Google Drive/docs/A_computational_theory_of_clustering/tables"
    RANDOM_SEED = random.randint(0, 10_000)


PAPER_DATASET_ORDER = [
    "1d_gauss",
    "2d_gauss",
    "6NewsgroupsUMAP10",
    "MNIST_UMAP10",
    "banknote",
    "breast_cancer",
    "cover_type",
    "glass",
    "ionosphere_UMAP10",
    "iris",
    "land_mines",
    "pendigits",
    "seeds",
    "shuttle",
    "wine",
    "yeast",
]


def plot_dataset_overview(dataset_name, df, feature_columns):
    if not Config.PLOT_FIGURES:
        return

    logger.info("Plotting dataset: %s", dataset_name)

    fig_true = plot_clusters(
        df,
        feature_columns,
        label_column='y_true',
        x_axis_label='',
        y_axis_label='Count',
        legend_label='Cluster Labels',
        title=f"{dataset_name} (Ground Truth)",
        show_seeds_only=False,
    )

    fig_seeds = plot_clusters(
        df,
        feature_columns,
        label_column='y_live',
        title=f"{dataset_name} (Seed Labels Only)",
        show_seeds_only=True,
    )

    if not Config.SAVE_PLOTS:
        return

    Config.PLOT_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    fig_true_path = Config.PLOT_SAVE_PATH / f"{dataset_name}_ytrue.png"
    fig_seeds_path = Config.PLOT_SAVE_PATH / f"{dataset_name}_ylive_seeds_only.png"

    fig_true.savefig(fig_true_path, dpi=300, bbox_inches='tight')
    fig_seeds.savefig(fig_seeds_path, dpi=300, bbox_inches='tight')
    logger.info("Saved plots to:\n- %s\n- %s", fig_true_path, fig_seeds_path)


def run_optional_visualizations(
    dataset_name,
    df,
    result_frames_by_method,
    feature_columns,
    plot_figures_dataset_specific,
):
    if Config.IS_TESTING and plot_figures_dataset_specific:
        plot_enabled_clusterings(
            df,
            clustering_flags,
            feature_columns,
            plot_save_path=Config.PLOT_SAVE_PATH,
            dataset_name=dataset_name,
            save_plots=Config.SAVE_PLOTS,
        )

    if Config.IS_TESTING:
        plot_confusion_matrices_for_clustering(
            df,
            true_label_col='y_true',
            clustering_flags=clustering_flags,
        )

    plot_embedding_label_views(
        dataset_name,
        result_frames_by_method,
        project_root=str(ROOT_PATH),
        plot_save_path=Config.PLOT_SAVE_PATH,
        save_plots=Config.SAVE_PLOTS,
    )


def build_output_tables(all_metrics):
    df_metrics = metrics_to_dataframe(all_metrics)
    if df_metrics.empty:
        return {}

    df_metrics["value"] = df_metrics["value"].round(4)

    dataset_order = PAPER_DATASET_ORDER
    df_median_metrics = median_metrics_dataframe(df_metrics)

    metric_tables = create_metric_tables(df_median_metrics, dataset_order=dataset_order)
    rebuttal_tables = {
        "full_eval_proposed": create_proposed_full_dataset_table(
            df_median_metrics,
            dataset_order=dataset_order,
        ),
        "full_eval_compare": create_full_comparison_table(
            df_median_metrics,
            dataset_order=dataset_order,
        ),
    }
    return {**metric_tables, **rebuttal_tables}


def display_output_tables(tables):
    for table_name, df in tables.items():
        display_table(table_name, df, display_fn=display)


def get_rebuttal_tables(tables):
    rebuttal_table_names = ("full_eval_proposed", "full_eval_compare")
    return {
        table_name: tables[table_name]
        for table_name in rebuttal_table_names
        if table_name in tables
    }


def save_output_tables(tables, save_pickle=True):
    if not tables:
        return

    Config.RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    save_metric_tables_as_latex(tables, Config.TABLE_SAVE_PATH)

    for table_name, df in tables.items():
        save_table_csv(table_name, df, Config.RESULTS_FOLDER)

    if save_pickle:
        with open(Config.RESULTS_FOLDER / "metric_tables.pkl", "wb") as f:
            pickle.dump(tables, f)


def emit_output_tables(tables):
    if not tables:
        return

    display_output_tables(tables)

    if Config.SAVE_RESULTS:
        save_output_tables(tables)
        return

    if Config.SAVE_REBUTTAL_RESULTS:
        save_output_tables(get_rebuttal_tables(tables), save_pickle=False)


# Initialise logger
logger = setup_logging(Config.IS_TESTING)

# -------------------------- Run Experiment -----------------------------------

# Set this to a specific dataset index to run only one dataset
SINGLE_DATASET_INDEX = None
dataset_indices = [SINGLE_DATASET_INDEX] if SINGLE_DATASET_INDEX is not None else list(dataset_dict.keys())

# holder for all metrics across datasets
all_metrics = {}

for dataset_index in dataset_indices:
    dataset_cfg = dataset_dict[dataset_index]

    # Resolve dataset parameters with fallbacks; default get is None
    dataset_name = dataset_cfg["name"]
    random_seed = dataset_cfg["random_seed"] if dataset_cfg.get("random_seed") is not None else Config.RANDOM_SEED
    plot_figures_dataset_specific = dataset_cfg.get("plot_figure", Config.PLOT_FIGURES)
    k = dataset_cfg.get("k")
    percent_labelled = dataset_cfg.get("percent_labelled")
    standardise = dataset_cfg.get("standardise", False)

    # load the dataset for plotting purposes and obtaining dataset characteristics
    df, num_clusters, plot_title, feature_columns = load_dataset(
        dataset_name, random_seed, k, percent_labelled, standardise,
    )

    # Save for later use
    number_of_examples = df.shape[0]
    number_of_seeds = (df['y_live'] != -1).sum()
    number_of_features = len(feature_columns)

    logger.info(
        f"Loaded dataset '{dataset_name}' with parameters:\n"
        f"  random_seed       = {random_seed}\n"
        f"  k                 = {k}\n"
        f"  percent_labelled  = {percent_labelled}\n"
        f"  standardise       = {standardise}\n"
        f"  Number of seeds   = {number_of_seeds}\n"
        f"  Number of examples= {number_of_examples}\n"
        f"  Number of features= {number_of_features}"
    )

    logger.debug("Class distribution (y_live):\n%s", df['y_live'].value_counts())

    # Plot dataset overview
    plot_dataset_overview(dataset_name, df, feature_columns)

    # Execute clustering algorithms based on the provided configurations and flags
    logger.info("******* Preparing to apply clustering methods for dataset %s *******" % dataset_name)

    profiler = start_profiler(Config.PROFILE_CODE)

    # Returns nested metric lists by dataset and one representative result frame per method for plotting.
    metrics_by_dataset, result_frames_by_method = run_metrics_time_clusterings(
        dataset_name = dataset_name,
        random_seed = Config.RANDOM_SEED,
        k = k,
        percent_labelled = percent_labelled,
        standardise = standardise,
        clustering_configs = clustering_configs,
        clustering_flags = clustering_flags,
        skip_clusterings = skip_clustering,
        num_repeats=3,
        load_dataset=load_dataset,
        selected_metrics=selected_metrics,
        num_examples = number_of_examples,
    )
    
    all_metrics[dataset_name] = metrics_by_dataset

    finish_profiler(profiler)

    logger.info("******* Completed running clustering and metrics for dataset %s for all methods *******" % dataset_name)

    # Optional post-run visualisations
    run_optional_visualizations(
        dataset_name,
        df,
        result_frames_by_method,
        feature_columns,
        plot_figures_dataset_specific,
    )

# %% Convert metrics to output tables and emit them 
all_output_tables = build_output_tables(all_metrics)
emit_output_tables(all_output_tables)
