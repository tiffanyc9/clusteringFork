# %%
# ---------------------------- Imports and Setup ----------------------------
import sys
import logging
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display

# Project Root Resolution
CURRENT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT_PATH = CURRENT_DIR.parent
sys.path.insert(0, str(ROOT_PATH))

# Internal Imports
from clustering_nassir.cluster import SemiSupervisedClusterer
from evaluation_configs import dataset_dict
from utilities.evaluation_metrics import compute_purity, compute_ari, compute_v_measure, compute_nmi, compute_fmi
from utilities.generate_load_data import load_dataset

# %%
# -------------------------- Experiment Configuration ------------------------

class Config:
    """Centralised configuration settings."""
    PROFILE_CODE = False
    IS_TESTING = False
    RESULTS_FOLDER = Path("results")
    PLOT_FIGURES = False
    SAVE_RESULTS = False
    SAVE_PLOTS = True
    PLOT_SAVE_PATH = Path.home() / "Google Drive/docs/A_computational_theory_of_clustering/figures"
    RANDOM_SEED = random.randint(0, 10_000) 

def setup_logging(is_testing: bool) -> logging.Logger:
    """Initialise and return logger with level based on testing mode."""

    if is_testing:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
            handlers=[logging.StreamHandler()]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
            handlers=[logging.StreamHandler()]
        )

    # Suppress matplotlib font warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logger = logging.getLogger("clustering")
    return logger

# Initialise project-wide logger
my_logger = setup_logging(Config.IS_TESTING)

# Define metrics to compute
metric_functions = {
    "Purity": compute_purity,
    "NMI": compute_nmi,
    "V_measure": compute_v_measure,  
    "ARI": compute_ari, 
    "FMI": compute_fmi,
}

# %% Loop over points per cluster directly
# ----------------------- Main Experiment Logic for Points per Cluster vs Clustering Metrics (using actual number of points) ------------------

points_per_cluster_list = [5, 10, 
                           15, 20, 25, 30, 
                           35, 40, 45, 50,
                           ]
results = []

for dataset_cfg in dataset_dict.values():
    dataset_name = dataset_cfg["name"]
    k = dataset_cfg.get("k", None)
    random_seed = Config.RANDOM_SEED
    standardise = dataset_cfg.get("standardise", False)

    if dataset_name == "cover_type":
        num_repeats = 3
    else:
        num_repeats = 10
            
    my_logger.info(f"Loading dataset '{dataset_name}' with random_seed={random_seed}")
    
    # Load once to get number of total points
    df_tmp, _, _, _ = load_dataset(
        dataset_name,
        random_seed,
        k,
        0.3,
        standardise=standardise,
    )
    num_total_points = df_tmp.shape[0]
    print(f"Starting main loop for dataset '{dataset_name}' with {num_total_points} total points.")
    
    for target_points_per_cluster in points_per_cluster_list:
        metrics_accumulator = defaultdict(list)

        for repeat in range(num_repeats):
            total_labelled = target_points_per_cluster * k
            proportion_labelled = total_labelled / num_total_points

            my_logger.info(f"total_labelled: {total_labelled}, k: {k}, num_total_points: {num_total_points}")
            my_logger.info(f"proportion_labelled: {proportion_labelled:.3f}")

            df, num_clusters, plot_title, feature_columns = load_dataset(
                dataset_name,
                random_seed + repeat,
                k,
                proportion_labelled,
                standardise,
            )

            labelled_counts = df[df['y_live'] != -1]['y_live'].value_counts().to_dict()
            if any(count < 3 for count in labelled_counts.values()):
                my_logger.warning(f"Skipping: Not enough seeds per cluster: {labelled_counts}")
                continue

            clf = SemiSupervisedClusterer()
            df["novel_method"] = clf.fit(df[feature_columns + ['y_live']].to_numpy())
            df_filtered = df[(df['y_true'] != -1) & (df['novel_method'] != -1)]

            for metric_name, metric_fn in metric_functions.items():
                score = metric_fn(df_filtered, true_col="y_true", pred_col="novel_method")
                metrics_accumulator[metric_name].append(score)

        if not metrics_accumulator:
            continue

        my_logger.info (f"purity scores: {metrics_accumulator['Purity']}")

        for metric_name, scores in metrics_accumulator.items():
            if not scores:
                continue

            avg_score = np.mean(scores)                  
            std_score = np.std(scores)                   
            results.append({
                "Dataset": dataset_name,
                "Percentage Labelled": proportion_labelled * 100,
                "Points per Cluster": target_points_per_cluster,
                "Metric": metric_name,
                "Score": avg_score,
                "Std": std_score,                         
                "Random Seed": random_seed
            })

        my_logger.info(f"Processed dataset '{dataset_name}' with {target_points_per_cluster} pts/cluster.")

# Convert to DataFrame
df_all = pd.DataFrame(results)

# ------------------------ Plotting ------------------------

Config.PLOT_SAVE_PATH.mkdir(parents=True, exist_ok=True)
logging.getLogger().setLevel(logging.WARNING)

for metric in df_all["Metric"].unique():
    df_metric = df_all[df_all["Metric"] == metric]

    fig = px.line(
        df_metric,
        x="Points per Cluster",
        y="Score",
        color="Dataset",
        markers=True,
        labels={"Score": metric},
        # title=f"{metric} vs Points per Cluster",
        hover_data=["Random Seed", "Dataset", "Points per Cluster", "Score", "Std"]
    )
    fig.update_layout(
        legend_title="Dataset",
        xaxis=dict(title="Points per Cluster"),
        yaxis=dict(title=metric, range=[0, 1.05]),
    )

    fig.show()

    if Config.SAVE_PLOTS:
        file = Config.PLOT_SAVE_PATH / f"{metric.lower()}_vs_points_per_cluster_all.png"
        fig.write_image(str(file))
        print(f"Saved plot: {file}")

        # Save interactive HTML
        html_file = Config.PLOT_SAVE_PATH / f"{metric.lower()}_vs_points_per_cluster_all.html"
        fig.write_html(str(html_file))
        print(f"Saved interactive HTML: {html_file}")

# %%
# # ----------------------- Main Experiment Logic for Points per Cluster vs Clustering Metrics ------------------

# proportion = [0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.1, 0.2, 0.3, 0.4]
# num_repeats = 10
# results = []

# # Main experiment loop
# for dataset_cfg in dataset_dict.values():
#     dataset_name = dataset_cfg["name"]
#     k = dataset_cfg.get("k", None)
#     random_seed = Config.RANDOM_SEED
#     standardise = dataset_cfg.get("standardise", False)

#     my_logger.info(f"Loading dataset '{dataset_name}' with random_seed={random_seed}")

#     for proportion_labelled in proportion:
#         metrics_accumulator = defaultdict(list)

#         # accumulate points per cluster to average over repeats
#         points_per_cluster_list = []

#         for repeat in range(num_repeats):
#             df, num_clusters, plot_title, feature_columns = load_dataset(
#                 dataset_name,
#                 random_seed + repeat,
#                 k,
#                 proportion_labelled,
#                 standardise,
#             )

#             labelled_counts = df[df['y_live'] != -1]['y_live'].value_counts().to_dict()
#             if any(count < 3 for count in labelled_counts.values()):
#                 my_logger.warning(f"Skipping: Not enough seeds per cluster: {labelled_counts}")
#                 continue

#             total_labelled = proportion_labelled * df.shape[0]
#             points_per_cluster = round(total_labelled / k)
#             points_per_cluster_list.append(points_per_cluster)

#             clf = NovelClustering()
#             df["novel_method"] = clf.fit(df[feature_columns + ['y_live']].to_numpy())
#             df_filtered = df[(df['y_true'] != -1) & (df['novel_method'] != -1)]

#             # Compute all metrics
#             for metric_name, metric_fn in metric_functions.items():
#                 score = metric_fn(df_filtered, "novel_method", "y_true")
#                 metrics_accumulator[metric_name].append(score)

#         if not metrics_accumulator:
#             continue

#         avg_points_per_cluster = round(sum(points_per_cluster_list) / len(points_per_cluster_list))

#         for metric_name, scores in metrics_accumulator.items():
#             if not scores:
#                 continue
#             avg_score = sum(scores) / len(scores)
#             results.append({
#                 "Dataset": dataset_name,
#                 "Percentage Labelled": proportion_labelled * 100,
#                 "Points per Cluster": avg_points_per_cluster,
#                 "Metric": metric_name,
#                 "Score": avg_score,
#                 "Random Seed": random_seed
#             })
            
#     my_logger.info(f"Processed dataset '{dataset_name}' with proportion {proportion_labelled:.3f}.")

# my_logger.info(f"Completed processing all datasets for points per cluster.")

# # Convert to DataFrame
# df_all = pd.DataFrame(results)

# # Ensure the save directory exists
# Config.PLOT_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# # Plot and save: Score vs Points per Cluster, grouped by Metric and Dataset
# for metric_name in metric_functions.keys():
#     df_metric = df_all[df_all["Metric"] == metric_name]
#     fig = px.line(
#         df_metric,
#         x="Points per Cluster",
#         y="Score",
#         color="Dataset",
#         markers=True,
#         labels={"Score": metric_name},
#         hover_data=["Random Seed", "Dataset", "Points per Cluster", "Score"]
#     )
#     fig.update_layout(
#         # title=f"{metric_name} vs Points per Cluster (All Datasets)",
#         legend_title="Dataset",
#         xaxis=dict(title="Points per Cluster", tickformat=".0f", range=[0, 100]),
#         yaxis=dict(title=metric_name, range=[0, 1.05]),
#     )
    
#     # Show plot (optional, remove if batch mode)
#     fig.show()

#     # Save plot
#     filename = Config.PLOT_SAVE_PATH / f"{metric_name.lower()}_vs_points_per_cluster.png"
#     fig.write_image(str(filename))
#     print(f"Saved: {filename}")



# %%
# ----------------------- Main Experiment Logic for Label Noise vs Metrics ------------------

# # Parameters
# noise_rates = [0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
#                0.6, 0.7, 0.8, 0.9, 1.0]

# proportion_labelled = 0.1 # fix at 10% labelled data
# num_repeats = 5
# results_with_noise = []

# for dataset_cfg in dataset_dict.values():
#     dataset_name = dataset_cfg["name"]
#     k = dataset_cfg.get("k", None)
#     random_seed = dataset_cfg.get("random_seed") or Config.RANDOM_SEED
#     standardise = dataset_cfg.get("standardise", False)

#     for noise_rate in noise_rates:
#         metric_accumulator = defaultdict(list)

#         for repeat in range(num_repeats):
#             df, num_clusters, plot_title, feature_columns = load_dataset(
#                 dataset_name,
#                 random_seed + repeat,
#                 k,
#                 proportion_labelled,
#                 standardise
#             )

#             # Corrupt labels
#             seed_mask = df['y_live'] != -1
#             seed_indices = df[seed_mask].index.tolist()
#             num_seeds_to_flip = int(len(seed_indices) * noise_rate)

#             if num_seeds_to_flip > 0:
#                 flip_indices = random.sample(seed_indices, num_seeds_to_flip)
#                 for idx in flip_indices:
#                     current_label = df.at[idx, 'y_live']
#                     all_labels = list(set(df['y_true'].unique()) - {-1})
#                     new_label = random.choice([l for l in all_labels if l != current_label])
#                     df.at[idx, 'y_live'] = new_label

#             # Fit clustering
#             clf = NovelClustering()
#             try:
#                 df["novel_method"] = clf.fit(df[feature_columns + ['y_live']].to_numpy())
#             except ValueError:
#                 continue

#             # Filter out outliers for metrics computation
#             df_filtered = df[(df['y_true'] != -1) & (df['novel_method'] != -1)]

#             # Compute metrics
#             for metric_name, metric_fn in metric_functions.items():
#                 score = metric_fn(df_filtered, "novel_method", "y_true")
#                 metric_accumulator[metric_name].append(score)

#         for metric_name, scores in metric_accumulator.items():
#             if scores:
#                 results_with_noise.append({
#                     "Dataset": dataset_name,
#                     "Noise Rate": noise_rate,
#                     "Proportion Labelled": proportion_labelled,
#                     "Metric": metric_name,
#                     "Average Score": round(np.mean(scores), 2),
#                     "Std Dev": round(np.std(scores), 2),
#                     "Num Repeats": len(scores)
#                 })

#     my_logger.info(f"Completed processing dataset '{dataset_name}' with noise rates.")

# # Convert to DataFrame
# results_with_noise_df = pd.DataFrame(results_with_noise)
# print(results_with_noise_df)

# # Ensure the save directory exists
# Config.PLOT_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# # Plot each metric
# for metric_name in metric_functions:
#     df_metric = results_with_noise_df[results_with_noise_df['Metric'] == metric_name]
#     fig = px.line(
#         df_metric,
#         x="Noise Rate",
#         y="Average Score",
#         color="Dataset",
#         markers=True,
#         error_y="Std Dev",
#         labels={"Average Score": metric_name},
#         hover_data=["Noise Rate", "Average Score", "Std Dev", "Dataset"]
#     )
#     fig.update_layout(
#         title=f"{metric_name} vs Label Noise Rate",
#         xaxis=dict(title="Label Noise Rate", tickvals=noise_rates),
#         yaxis=dict(title=metric_name, range=[0, 1.05]),
#         legend_title="Dataset"
#     )

#     # Show plot (optional, remove if batch mode)
#     fig.show()

#     filename = Config.PLOT_SAVE_PATH / f"{metric_name.lower()}_vs_label_noise_rate.png"
#     fig.write_image(str(filename))
#     print(f"Saved: {filename}")

#     # Optionally save to CSV
#     # results_df.to_csv("purity_vs_label_noise.csv", index=False)


# %% test code

# dataset_cfg in [dataset_dict[5]]
# dataset_name = dataset_cfg["name"]
# k = dataset_cfg.get("k", None)
# # Use random seed from config or default if None or missing
# random_seed = dataset_cfg.get("random_seed") #or Config.RANDOM_SEED
# standardise = dataset_cfg.get("standardise", False)
# percent_labelled = dataset_cfg.get("percent_labelled", None)

# df, num_clusters, plot_title, feature_columns = load_dataset(
#         dataset_name,
#         random_seed,
#         k,
#         percent_labelled,
#         standardise,
#     )

# from clustering_methods import novel_clustering
# df_o = novel_clustering(df, feature_columns, target_column='y_true', 
#                         seeds='y_live', remap_labels=False)
    
# # Filter for outliers (only once per method)
# df_filtered = df_o[(df_o['y_true'] != -1) & (df_o['novel_method'] != -1)]

# purity = compute_purity(df_filtered, "novel_method", "y_true")
# print(purity)

# %% experiments - Future Work

# # Replace this with the dataset config you're testing
# dataset_cfg = dataset_dict[1]
# dataset_name = dataset_cfg["name"]
# k = dataset_cfg.get("k", None)
# random_seed = dataset_cfg.get("random_seed") or Config.RANDOM_SEED
# standardise = dataset_cfg.get("standardise", False)
# percent_labelled = 0.1 #dataset_cfg.get("percent_labelled", None)

# # Load dataset
# df, num_clusters, plot_title, feature_columns = load_dataset(
#     dataset_name,
#     random_seed,
#     k,
#     percent_labelled,
#     standardise,
# )

# # Assign completely random labels to y_live, uniformly from 0 to k-1
# np.random.seed(random_seed)
# df["y_live"] = np.random.choice(k, size=len(df))

# # Run your clustering method
# df_o = novel_clustering(df, feature_columns, target_column='y_true', 
#                         seeds='y_live', remap_labels=False)

# # Remove outliers
# df_filtered = df_o[(df_o['y_true'] != -1) & (df_o['novel_method'] != -1)]

# # Compute and print purity
# purity = compute_purity(df_filtered, "novel_method", "y_true")
# print(f"Purity with uniform random labels (no knowledge of y_true): {purity:.4f}")

# # %%

# from collections import defaultdict
# import numpy as np
# import random

# # Configuration
# dataset_cfg = dataset_dict[2]  # Replace with desired dataset index
# dataset_name = dataset_cfg["name"]
# k = dataset_cfg.get("k", None)
# random_seed = dataset_cfg.get("random_seed") or Config.RANDOM_SEED
# standardise = dataset_cfg.get("standardise", False)
# proportion_labelled = 0.1

# # Load dataset with 10% labelled
# df, num_clusters, plot_title, feature_columns = load_dataset(
#     dataset_name,
#     random_seed,
#     k,
#     proportion_labelled,
#     standardise
# )

# # Replace all seed labels with randomly selected labels (could be same as original)
# np.random.seed(random_seed)
# random.seed(random_seed)

# seed_mask = df['y_live'] != -1
# seed_indices = df[seed_mask].index.tolist()

# true_labels = sorted(set(df['y_true'].unique()) - {-1})
# for idx in seed_indices:
#     noisy_label = random.choice(true_labels)
#     df.at[idx, 'y_live'] = noisy_label

# # Apply clustering
# clf = NovelClustering()
# df["novel_method"] = clf.fit(df[feature_columns + ['y_live']].to_numpy())

# # Evaluate
# df_filtered = df[(df['y_true'] != -1) & (df['novel_method'] != -1)]
# purity = compute_purity(df_filtered, "novel_method", "y_true")
# print(f"Purity after randomly reassigning 10% seed labels: {purity:.4f}")



