"""
Created on Mon Feb  6 13:05:33 2023

@author: nassirmohammad
"""
import os
import logging
import numpy as np
import pandas as pd

METHOD_ORDER = [
    'KMeans',
    'GMM',
    'SeededKMeans',
    'ConstrainedKMeans',
    'COPKMeans',
    'Agglomerative',
    'DBSCAN',
    'DEC',
    'novel_method',
]

METHOD_NAME_MAP = {
    'Agglomerative': 'Agg',
    'SeededKMeans': 'S-KM',
    'ConstrainedKMeans': 'C-KM',
    'COPKMeans': 'COPKM',
    'novel_method': 'Proposed',
}

# Define save_df helper
def save_df(df, filename_prefix, dataset_name, results_folder):
    filename = os.path.join(results_folder, f"{filename_prefix}_{dataset_name}.csv")
    df.to_csv(filename, index=False)
    logging.info(f"{filename_prefix.replace('_', ' ').capitalize()} saved to {filename}")

def _table_csv_filename(table_name):
    return (
        table_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    ) + ".csv"

def style_table(table_name, df):
    """Return a styled table for notebook/script display."""
    if df.empty:
        return None

    cmap = "Reds_r" if "runtime (s)" in table_name.lower() else "Greens"
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    def formatter(value):
        if pd.isna(value):
            return "--"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:.2f}"
        return str(value)

    styled = df.style.format(formatter).set_caption(table_name)
    if len(numeric_columns) > 0:
        styled = styled.background_gradient(cmap=cmap, subset=numeric_columns, axis=1)

    return styled

def display_table(table_name, df, display_fn):
    """Display a table if it is non-empty."""
    styled = style_table(table_name, df)
    if styled is not None:
        display_fn(styled)

def save_table_csv(table_name, df, results_folder):
    """Save a table as CSV if it is non-empty."""
    if df.empty or results_folder is None:
        return

    os.makedirs(results_folder, exist_ok=True)
    df.to_csv(os.path.join(results_folder, _table_csv_filename(table_name)), index=True)

def display_and_save_table(table_name, df, display_fn, save_results=False, results_folder=None):
    """Convenience wrapper that displays a table and optionally saves it as CSV."""
    if df.empty:
        return

    display_table(table_name, df, display_fn)

    if save_results and results_folder is not None:
        save_table_csv(table_name, df, results_folder)

def combine_results(results_folder="results"):
    runtime_files = []
    metrics_files = []

    # Scan folder for files
    for filename in os.listdir(results_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_folder, filename)
            # Heuristic: runtime files only have 3 columns
            with open(filepath, 'r') as f:
                header = f.readline()
                num_columns = len(header.strip().split(","))
                if num_columns == 3 and "Runtime" in header:
                    runtime_files.append(filepath)
                elif num_columns > 3:
                    metrics_files.append(filepath)

    # Read and combine runtimes
    runtime_dfs = [pd.read_csv(file) for file in runtime_files]
    all_runtimes = pd.concat(runtime_dfs, ignore_index=True) if runtime_dfs else pd.DataFrame()

    # Read and combine metrics
    metrics_dfs = [pd.read_csv(file) for file in metrics_files]
    all_metrics = pd.concat(metrics_dfs, ignore_index=True) if metrics_dfs else pd.DataFrame()

    if all_metrics.empty:
        print("⚠️ No metrics files found.")
        return pd.DataFrame()  # return empty DataFrame

    if all_runtimes.empty:
        print("⚠️ No runtime files found. The result will only contain metrics.")

    # Merge on Algorithm and Dataset (left join metrics with runtimes)
    combined = pd.merge(all_metrics, all_runtimes, on=["Algorithm", "Dataset"], how="left") if not all_runtimes.empty else all_metrics

    # Optional: sort for readability
    combined = combined.sort_values(by=["Dataset", "Algorithm"]).reset_index(drop=True)

    print("✅ Combined results ready.")
    return combined

def escape_latex(s):
    if not isinstance(s, str):
        return s
    return (s.replace('\\', '\\textbackslash{}')
             .replace('&', '\\&')
             .replace('%', '\\%')
             .replace('$', '\\$')
             .replace('#', '\\#')
             .replace('_', '\\_')  # ← this is the one causing your error
             .replace('{', '\\{')
             .replace('}', '\\}')
             .replace('~', '\\textasciitilde{}')
             .replace('^', '\\textasciicircum{}'))

def process_df(df_cr):
    # round values
    df_cr = df_cr.round(2)

    # clean up dataset names (probably not needed anymore)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_with_class", "", regex=False)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_class", "", regex=False)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_trn", "", regex=False)
    df_cr["Dataset"] = df_cr["Dataset"].str.replace("_txt", "", regex=False)

    rename_algorithms = {
        "KMeans": "k-means",
        "DBSCAN": "DBSCAN",
        "Agglomerative": "Agg",
        "GaussianMixture": "GMM",
        "ConstrainedKMeans": "C-KM",
        "SeededKMeans": "S-KM",
        "novel_method": "Ours",
        "COPKMeans": "COPKM",
    }

    # filter and rename algorithms
    methods_to_remove = {"Spectral", "MeanShift"}
    df_cr = df_cr[~df_cr["Algorithm"].isin(methods_to_remove)]
    df_cr["Algorithm"] = df_cr["Algorithm"].replace(rename_algorithms)

    # metrics to keep
    metrics = ['Purity', 'V-Measure', 'NMI', 'ARI', 'FMI', 'Runtime (s)']

    metric_dfs = {}
    for metric in metrics:
        metric_df = df_cr.pivot(index="Dataset", columns="Algorithm", values=metric)

        # Move 'Ours' to the end
        cols = list(metric_df.columns)
        if "Ours" in cols:
            cols.remove("Ours")
            cols.append("Ours")
            metric_df = metric_df[cols]

        # Escape dataset names (index)
        metric_df.index = metric_df.index.map(escape_latex)
        metric_dfs[metric] = metric_df

    return metric_dfs

def metrics_to_dataframe(all_metrics):
    """Flatten nested metrics structure into a DataFrame."""
    records = [
        {
            "dataset": dataset_name,
            "metric": metric,
            "method": method,
            "repeat": i + 1,
            "value": value
        }
        for dataset_name, inner_dict in all_metrics.items()
        for metric, methods in inner_dict[dataset_name].items()
        for method, values in methods.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(records)

def average_metrics_dataframe(df):
    """Compute average of each metric per method and dataset."""
    return df.groupby(['dataset', 'metric', 'method'], as_index=False)['value'].mean()

def median_metrics_dataframe(df):
    return df.groupby(['dataset', 'metric', 'method'], as_index=False)['value'].median()

def escape_latex_underscores(text: str) -> str:
    return text.replace('_', r'\_')

def _ordered_index(values, preferred_order):
    if not preferred_order:
        return list(values)

    preferred = [value for value in preferred_order if value in values]
    remainder = [value for value in values if value not in preferred]
    return preferred + remainder

def _dataset_sizes_from_metrics(df_avg):
    dataset_sizes = (
        df_avg[df_avg['metric'] == 'Dataset Size']
        .drop_duplicates(subset=['dataset'])
        .set_index('dataset')['value']
        .to_dict()
    )
    return {dataset: int(round(size)) for dataset, size in dataset_sizes.items() if pd.notna(size)}

def _pivot_metric(df_avg, metric_name, methods_to_include=None, dataset_order=None):
    df_metric = df_avg[df_avg['metric'] == metric_name].copy()
    if methods_to_include is not None:
        df_metric = df_metric[df_metric['method'].isin(methods_to_include)]

    if df_metric.empty:
        return pd.DataFrame()

    pivot = df_metric.pivot(index='dataset', columns='method', values='value')

    if methods_to_include is not None:
        for method_name in methods_to_include:
            if method_name not in pivot.columns:
                pivot[method_name] = pd.NA
        pivot = pivot[methods_to_include]

    pivot = pivot.reindex(_ordered_index(pivot.index, dataset_order))
    return pivot

def create_metric_tables(df_avg, dataset_order=None):
    """Create pivot tables per metric."""
    tables = {}
    metrics_to_include = {
        'Purity',
        'V-Measure',
        'NMI',
        'ARI',
        'FMI',
        'runtime (s)',
    }

    methods_to_include = [
        'KMeans',
        'GMM',
        'SeededKMeans',
        'ConstrainedKMeans',
        'COPKMeans',
        'Agglomerative',
        'novel_method',
        'DBSCAN',
        'DEC',
    ]

    for metric, df_metric in df_avg.groupby('metric'):
        if metric not in metrics_to_include:
            continue

        df_metric = df_metric.copy()
        df_metric = df_metric[df_metric['method'].isin(methods_to_include)]

        pivot = df_metric.pivot(index='dataset', columns='method', values='value')

        cols = [m for m in methods_to_include if m in pivot.columns and m != 'novel_method']
        if 'novel_method' in pivot.columns:
            cols.append('novel_method')
        pivot = pivot[cols]
        pivot = pivot.reindex(_ordered_index(pivot.index, dataset_order))

        pivot.columns = [METHOD_NAME_MAP.get(col, col) for col in pivot.columns]

        pivot.index = pivot.index.map(escape_latex_underscores)
        pivot.columns = [escape_latex_underscores(col) for col in pivot.columns]

        tables[metric] = pivot

    return tables

def create_proposed_full_dataset_table(df_avg, dataset_order=None, method='novel_method'):
    """Create the proposed-method full-dataset evaluation table."""
    metric_order = [
        'ARI (full)',
        'NMI (full)',
        'V-Measure (full)',
        'FMI (full)',
        'Purity (full)',
        'Rejected Count',
        'Rejection Rate',
    ]
    dataset_sizes = _dataset_sizes_from_metrics(df_avg)

    df_metric = df_avg[df_avg['method'] == method].copy()
    df_metric = df_metric[df_metric['metric'].isin(metric_order)]

    if df_metric.empty:
        return pd.DataFrame()

    pivot = df_metric.pivot(index='dataset', columns='metric', values='value')

    for metric in metric_order:
        if metric not in pivot.columns:
            pivot[metric] = pd.NA

    pivot = pivot[metric_order]
    pivot = pivot.reindex(_ordered_index(pivot.index, dataset_order))

    rejected_strings = []
    for dataset_name, row in pivot.iterrows():
        rejected_count = row['Rejected Count']
        total_count = dataset_sizes.get(dataset_name)

        if pd.isna(rejected_count) or total_count in (None, 0):
            rejected_strings.append('--')
            continue

        rejected_strings.append(f"{int(round(rejected_count))}/{int(total_count)}")

    pivot['Rejected (n/N)'] = rejected_strings
    pivot = pivot.drop(columns=['Rejected Count'])
    pivot = pivot[
        [
            'ARI (full)',
            'NMI (full)',
            'V-Measure (full)',
            'FMI (full)',
            'Purity (full)',
            'Rejected (n/N)',
            'Rejection Rate',
        ]
    ]
    pivot = pivot.rename(
        columns={
            'ARI (full)': 'ARI',
            'NMI (full)': 'NMI',
            'V-Measure (full)': 'V-Measure',
            'FMI (full)': 'FMI',
            'Purity (full)': 'Purity',
        }
    )

    pivot.index = pivot.index.map(escape_latex_underscores)
    return pivot

def create_full_comparison_table(df_avg, dataset_order=None):
    """Create a compact full-dataset comparison table for ARI and NMI."""
    metric_blocks = []
    display_metric_names = {
        'ARI (full)': 'ARI',
        'NMI (full)': 'NMI',
    }

    for metric_name in ['ARI (full)', 'NMI (full)']:
        pivot = _pivot_metric(
            df_avg,
            metric_name,
            methods_to_include=METHOD_ORDER,
            dataset_order=dataset_order,
        )
        if pivot.empty:
            continue

        pivot.columns = [escape_latex_underscores(METHOD_NAME_MAP.get(col, col)) for col in pivot.columns]
        pivot.index = pd.MultiIndex.from_tuples(
            [
                (display_metric_names.get(metric_name, metric_name), escape_latex_underscores(dataset_name))
                for dataset_name in pivot.index
            ],
            names=['Metric', 'Dataset'],
        )
        metric_blocks.append(pivot)

    if not metric_blocks:
        return pd.DataFrame()

    return pd.concat(metric_blocks)

def save_metric_tables_as_latex(tables, save_path):
    """Save pivot tables as LaTeX files."""
    os.makedirs(save_path, exist_ok=True)
    special_filenames = {
        'full_eval_proposed': 'full_eval_proposed',
        'full_eval_compare': 'full_eval_compare',
    }

    for metric, pivot in tables.items():
        safe_name = special_filenames.get(metric, metric.replace(' ', '_').replace('_', '-'))
        file_path = os.path.join(save_path, f"{safe_name}.tex")
        with open(file_path, 'w') as f:
            f.write(pivot.to_latex(float_format="%.2f", na_rep='--'))
