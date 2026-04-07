# %% Imports
import os

import pandas as pd
import umap
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

""" We use the following datasets:
- Iris Data: n: 150, d: 4, classes: 3
             A classic and very clean dataset.
             It's small, easy to visualize, and has relatively well-separated clusters,
             making it ideal for initial testing and demonstrating basic algorithm functionality.
- Wine Data: n: 178, d: 13, classes: 3
             Another popular choice for its clear cluster structure and slightly higher
             dimensionality than Iris, providing a good step up in complexity.
- Breast Cancer Data: n: 569, d: 30, classes: 2
             A real-world medical dataset. While binary, it's often used to test how well clustering
             algorithms can separate these two important groups based on various diagnostic features.
- Seed Data: n: 210, d: 7, classes: 3
             A clean and straightforward dataset with clear cluster definitions, making it suitable
             for comparing the core performance of clustering algorithms.
- Glass Data: n: 214, d: 9, classes: 6
             Offers a higher number of classes, which can test an algorithm's ability to distinguish
             more granular groups. Some classes might be less separable, providing a moderate challenge.
- Ionosphere Data: n: 351, d: 34, classes: 2
             A more complex dataset with a higher number of features and a binary classification. As one
             class is good, and the other is bad, the bad class may not form a cohere cluster.
- Shuttle Data: n: 49097, d: 9, classes: 7
             A large dataset with a significant number of samples and features, With 58,000 instances,
             it serves as an excellent benchmark for assessing the computational efficiency and
             scalability of clustering algorithms. Algorithms that perform well on smaller datasets
             might struggle or be very slow on this larger volume of data.
             Class 5, 5% of data is all predicted as anomalies, what is an anomaly here, is the small
             group of instances another group or an anomoalous group? Ground truth can become blurry here.
- Yeast Data: n: 1484, d: 8, classes: 10
             A larger dataset with more classes, allowing for evaluation on more complex and potentially
             imbalanced clustering scenarios. # good example for failure analysis as methods do not perform well?
- Banknotes Data: n: 1372, d: 5, classes: 2
            Appears that there are more than 2 clusters, ground truth not clear, 5 clusters? Not used further.
- Pendigits Data: n: 7494, d: 16, classes: 10
             A larger dataset with more classes, allowing for evaluation on more complex and potentially
             imbalanced clustering scenarios. The dataset consists of handwritten digits, which can be
             challenging for clustering algorithms due to the variability in handwriting styles.
- CovType Data: n: 581012, d: 54, classes: 7
                A very large dataset with a high number of features and classes, suitable for testing the
                scalability and performance of clustering algorithms. The dataset is large and complex,
                making it suitable for benchmarking algorithms on high-dimensional data.
- MNIST Data: n: 60000, d: 784, classes: 10
             A classic dataset for handwritten digit recognition, with a large number of samples and high
             dimensionality. It serves as a benchmark for clustering algorithms, especially in terms of
             scalability and performance on high-dimensional data.
             The dataset is large and complex, making it suitable for testing the scalability and efficiency
             of clustering algorithms. The high dimensionality and variability in digit representation
             can challenge clustering methods, especially those that rely on distance metrics.
-20Newsgroups Data: n: 11314, d: 1000, classes: 20
             A text dataset with 20 different newsgroups, providing a challenge for clustering algorithms
             due to the high dimensionality and the need for effective text representation.
- Fashion MNIST Data: n: 60000, d: 784, classes: 10
             A dataset of fashion items, similar in structure to MNIST but with different classes. It provides
             a more complex clustering challenge due to the variety of clothing items and their features.
             The dataset is large and complex, making it suitable for testing the scalability and efficiency
             of clustering algorithms. The high dimensionality and variability in clothing item representation
             can challenge clustering methods, especially those that rely on distance metrics.
"""


# %% Paths
CURRENT_DIR = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(ROOT_PATH, "data", 'raw', "tabular")
SAVE_DATA_DIR = os.path.join(ROOT_PATH, "data", 'processed')


def _raw_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def _save_path(filename: str) -> str:
    return os.path.join(SAVE_DATA_DIR, filename)


def _save_df(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(_save_path(filename), index=False)


def _rename_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    num_features = df.shape[1] - 1
    df.columns = [f"f{i+1}" for i in range(num_features)] + ['class']
    return df


def _move_class_to_end(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in df.columns if col != 'class'] + ['class']
    return df[columns]


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]


def _build_umap_dataframe(X, y, n_components: int, random_state=None) -> pd.DataFrame:
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    X_umap = reducer.fit_transform(X)
    umap_columns = [f"umap_dim_{i+1}" for i in range(n_components)]
    df_umap = pd.DataFrame(X_umap, columns=umap_columns)
    df_umap['class'] = y
    return df_umap


# %% Function: Load and Rename Class Column
def load_and_rename_class_column(filename: str, class_column: str) -> pd.DataFrame:
    """
    Load a CSV file from data/tabular and rename a given class column to 'class'.

    Parameters:
    - filename (str): Name of the CSV file (e.g., "Seed_Data.csv")
    - class_column (str): Column to be renamed to 'class'

    Returns:
    - pd.DataFrame: DataFrame with standardised 'class' column
    """
    csv_path = _raw_path(filename)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found. Available columns: {df.columns.tolist()}")

    return df.rename(columns={class_column: 'class'})


# %% ------------- process the shuttle dataset -------------
df = pd.read_csv(_raw_path("shuttle_trn.csv"), header=None, sep=r'\s+')
num_columns = df.shape[1]
df.columns = [f"f{i+1}" for i in range(num_columns - 1)] + ['original_class']
outlier_labels = {2, 3, 6, 7}
df['class'] = df['original_class'].apply(lambda x: -1 if x in outlier_labels else x)
df.drop(columns=['original_class'], inplace=True)
print(df.head())
_save_df(df, "shuttle_trn_with_class.csv")


# %% ------------- process the iris dataset -------------
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=[f"f{i+1}" for i in range(iris.data.shape[1])])
df_iris['class'] = iris.target
print(df_iris.head())
_save_df(df_iris, "iris_with_class.csv")


# %% ------------- process the ionosphere dataset -------------
df = pd.read_csv(_raw_path("ionosphere.csv"), header=None)
num_columns = df.shape[1]
df.columns = [f"f{i+1}" for i in range(num_columns - 1)] + ['class']
df['class'] = df['class'].map({'g': 1, 'b': 0})
assert df['class'].nunique() == 2, "Unexpected number of unique class values"
print(df.head())
_save_df(df, "ionosphere_with_class.csv")


# %% ------------- process the ionosphere dataset using UMAP -------------
df = pd.read_csv(_raw_path("ionosphere.csv"), header=None)
num_columns = df.shape[1]
df.columns = [f"f{i+1}" for i in range(num_columns - 1)] + ['class']
df['class'] = df['class'].map({'g': 1, 'b': 0})
assert df['class'].nunique() == 2, "Unexpected number of unique class values"

X = df.drop(columns=['class']).values
y = df['class'].values
df_umap = _build_umap_dataframe(X, y, n_components=10, random_state=42)
print(df_umap.head())
_save_df(df_umap, "ionosphere_umap10_with_class.csv")


# %% ------------- process the breast cancer dataset -------------
df = pd.read_csv(_raw_path("breast_cancer.csv"))
df = _drop_unnamed_columns(df)
df.drop(columns=["id"], inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df.rename(columns={"diagnosis": "class"}, inplace=True)
df = _move_class_to_end(df)
_save_df(df, "breast_cancer_class.csv")


# %% UMAP processing for breast cancer dataset
df = pd.read_csv(_raw_path("breast_cancer.csv"))
df = _drop_unnamed_columns(df)
df.drop(columns=["id"], inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df.rename(columns={"diagnosis": "class"}, inplace=True)
df = _move_class_to_end(df)

X = df.drop(columns=['class']).values
y = df['class'].values
df_umap = _build_umap_dataframe(X, y, n_components=20, random_state=42)
_save_df(df_umap, "breast_cancer_umap20_class.csv")


# %% ------------- process the wine dataset -------------
df = pd.read_csv(_raw_path("wine.csv"), header=None)
df['class'] = df[0]
df.drop(columns=[0], inplace=True)
df = _rename_feature_columns(df)
print(df.head())
_save_df(df, "wine_with_class.csv")


# %% ------------- process the glass dataset -------------
df = pd.read_csv(_raw_path("glass.csv"), header=None)
df.drop(columns=[0], inplace=True)
df['class'] = df[df.columns[-1]]
df.drop(columns=[df.columns[-1]], inplace=True)
df = _rename_feature_columns(df)
print(df.head())
_save_df(df, "glass_with_class.csv")


# %% ------------- process the yeast dataset -------------
df = pd.read_csv(_raw_path("yeast.csv"), header=None, delim_whitespace=True)
df.drop(columns=[0], inplace=True)
df.columns = list(df.columns[:-1]) + ['class']
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])
df = _rename_feature_columns(df)
small_classes = [1, 2, 3, 4, 8, 9]
df.loc[df['class'].isin(small_classes), 'class'] = -1
_save_df(df, "yeast_with_class.csv")


# %% process covtype dataset
df = pd.read_csv(_raw_path("covtype.csv"))
df = _drop_unnamed_columns(df)
if 'Cover_Type' in df.columns:
    df.rename(columns={'Cover_Type': 'class'}, inplace=True)

if 'class' in df.columns:
    df = _move_class_to_end(df)

X = df.drop(columns=['class']).values
y = df['class'].values
df_umap = _build_umap_dataframe(X, y, n_components=10)
_save_df(df_umap, "covtype_umap10_with_class.csv")
