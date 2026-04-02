# Seed-Guided Semi-Supervised Clustering

This repository contains the reference implementation for a semi-supervised clustering method that grows seed-defined clusters using anomaly detection. The method is described in:

- [Seed-Guided Semi-Supervised Clustering by A-Contrario Anomaly Detection](https://arxiv.org/abs/2306.06974)

At a high level, the algorithm starts from a small number of labelled seed points per class, fits an anomaly detector to each seeded cluster, ejects inconsistent points, and then attempts to claim compatible unlabelled points. Points that do not fit any seeded cluster remain labelled as `-1`.

## What The Package Does

- Accepts numeric feature data plus one seed-label column
- Uses `-1` as the unlabelled / anomalous label
- Iteratively refines seeded clusters rather than forcing every point into a cluster
- Returns a label for every input row, with rejected points preserved as `-1`

As a practical rule of thumb, the method usually benefits from at least 10 to 30 labelled examples per known class.

## Repository Layout

- [`clustering_nassir/`](clustering_nassir): installable Python package
- [`notebooks/`](notebooks): usage guides and exploratory analysis
- [`evaluation/`](evaluation): benchmark scripts and experiment drivers
- [`utilities/`](utilities): dataset loading, plotting, metrics, and helper functions
- [`tests/`](tests): test code

## Installation

There are two common ways to work with this repository.

### 1. Install The Core Package

Use this if you only want the clustering model itself.

```bash
pip install clustering-nassir
```

Or from a local checkout:

```bash
pip install .
```

### 2. Install The Full Research Environment

Use this if you want to run the notebooks, evaluation scripts, and paper experiments.

<!-- ### macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Windows
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
``` -->

### Conda environment
```bash
conda create --name clustering python=3.11
conda activate clustering
pip install -r requirements.txt
pip install -e .
```

The repository-level `requirements.txt` includes the extra scientific and notebook dependencies used by the evaluation code. The package `setup.py` remains focused on the core library dependency set.

## Quick Start

```python
import pandas as pd
from clustering_nassir import SemiSupervisedClusterer

df = pd.DataFrame(
    {
        "x1": [0.0, 0.2, 0.1, 4.9, 5.0, 5.2, 9.0],
        "x2": [0.1, 0.0, 0.3, 5.1, 4.8, 5.0, 9.1],
        "y_live": [0, 0, -1, 1, 1, -1, -1],
    }
)

X_with_seeds = df[["x1", "x2", "y_live"]].to_numpy()

model = SemiSupervisedClusterer(max_n_iterations=1000)
df["cluster"] = model.fit(X_with_seeds)

print(df)
```

### Input Format

- All feature columns must be numeric
- The final column must contain integer seed labels
- Use `-1` for unlabelled points
- Each labelled cluster must contain at least 3 seed examples

## Running The Notebooks

The easiest entry points are:

- [`notebooks/getting_started_guide_1.ipynb`](notebooks/getting_started_guide_1.ipynb)
- [`notebooks/getting_started_guide_2.ipynb`](notebooks/getting_started_guide_2.ipynb)
- [`notebooks/clustering_methods_and_evaluation.ipynb`](notebooks/clustering_methods_and_evaluation.ipynb)

Launch Jupyter from the repository root after installing the full environment:

```bash
jupyter lab
```

## Running The Evaluation Scripts

The main experiment entry point is:

- [`evaluation/clustering_evaluation.py`](evaluation/clustering_evaluation.py)

Typical usage:

```bash
python evaluation/clustering_evaluation.py
```

The datasets, enabled baselines, and selected metrics are configured in:

- [`evaluation/evaluation_configs.py`](evaluation/evaluation_configs.py)

## Development Notes

- The installable package is intentionally lightweight
- The evaluation code depends on a wider research stack than the core package
- Results, generated tables, and local experiment outputs are excluded by `.gitignore`
- Some benchmark scripts assume local datasets or external packages that are not required for the core clustering API

## Packaging

This repository now includes:

- `setup.py` for package metadata and installation
- `pyproject.toml` for modern build tooling
- `requirements.txt` for the full research environment

That combination is enough for local development, editable installs, and package builds. You do not need a separate `setup.cfg` unless you want to migrate more metadata out of `setup.py`.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
