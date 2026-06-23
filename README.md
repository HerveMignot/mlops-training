# MLOps Training

A hands-on MLOps training project covering the machine learning lifecycle from data preparation to model deployment.

## Setup

```bash
uv sync
```

## Project Structure

```
src/training/       Python package (data loading, features, model training)
notebooks/          Jupyter notebooks for interactive exercises
tests/              Unit tests
data/               Datasets (not tracked in git)
models/             Trained model artifacts (not tracked in git)
```

## Usage

```bash
uv run jupyter notebook    # Start Jupyter for interactive work
uv run pytest              # Run tests
uv run ruff check src/     # Lint code
```
