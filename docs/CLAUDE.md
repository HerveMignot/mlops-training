# Setting Up and Implementing MLOps Projects

Guide for building MLOps projects using the tools and patterns from this repository.

## Tool Stack

| Layer | Tool | Role |
|-------|------|------|
| Python packaging | **uv** | Dependency management, virtual environments, script execution |
| Build backend | **hatchling** | Builds the Python package from `src/` layout |
| ML framework | **scikit-learn** | Model training, pipelines, preprocessing |
| Data handling | **pandas** | Data loading, manipulation, feature engineering |
| Experiment tracking | **MLflow** | Experiment logging, model registry, artifact storage |
| Data versioning | **DVC** | Track large data files and define reproducible pipelines |
| API serving | **FastAPI** + **uvicorn** | REST API for model inference |
| Frontend | **Streamlit** | Interactive UI for predictions |
| Linting & formatting | **ruff** | Fast linter and formatter (replaces flake8, black, isort) |
| Type checking | **mypy** | Static type analysis |
| Git hooks | **pre-commit** | Run quality checks before every commit |
| Testing | **pytest** | Unit and integration tests |

## Project Structure

```
project/
├── pyproject.toml              # Single source for metadata, deps, tool config
├── .python-version             # Pin the Python version
├── .pre-commit-config.yaml     # Hooks: ruff + mypy
├── src/<package>/              # Source code (src layout)
│   ├── data.py                 # Data loading and preparation
│   ├── features.py             # Feature engineering (sklearn pipelines)
│   ├── train.py                # Training script with MLflow tracking
│   └── model.py                # Model training and evaluation functions
├── src/api/                    # Model serving
│   ├── app.py                  # FastAPI application
│   └── config.yaml             # Model path, port, etc.
├── src/frontend/               # User interface
│   ├── app.py                  # Streamlit application
│   └── config.yaml             # API endpoint URL
├── notebooks/                  # Jupyter notebooks for exploration
├── tests/                      # pytest test suite
├── data/                       # Datasets (tracked by DVC, not Git)
└── models/                     # Trained model artifacts (not in Git)
```

## Step-by-Step Setup

### 1. Initialize the project

```bash
uv init <project-name> --python 3.12
cd <project-name>
git init
```

Create the `src/` layout with `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "<package>"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "scikit-learn",
    "pandas",
    "mlflow==3.13",
]

[dependency-groups]
dev = ["pytest", "ruff", "mypy", "pre-commit"]

[tool.hatch.build.targets.wheel]
packages = ["src/<package>"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py312"
```

```bash
uv sync
```

### 2. Set up code quality

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.18
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v2.1.0
    hooks:
      - id: mypy
```

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### 3. Data versioning with DVC

```bash
uv add dvc
dvc init
```

Track data files:

```bash
dvc add data/<dataset>.csv
git add data/<dataset>.csv.dvc data/.gitignore
```

Optional — define a reproducible pipeline in `dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: python -m <package>.data
    outs:
      - data/<dataset>.csv
  train:
    cmd: python -m <package>.train
    deps:
      - data/<dataset>.csv
      - src/<package>/train.py
    outs:
      - models/model.joblib
```

```bash
dvc repro
```

### 4. Feature engineering with sklearn pipelines

Use `ColumnTransformer` + `Pipeline` to keep preprocessing reproducible and serializable with the model:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

def build_features_pipeline(categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), categorical_features),
            ("num", "passthrough", numerical_features),
        ],
    )
    return Pipeline([("preprocessor", preprocessor)])
```

Embed this in the training pipeline so the entire transform + model is saved as a single artifact.

### 5. Experiment tracking with MLflow

Start the tracking server:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

In the training script:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("<experiment-name>")
mlflow.sklearn.autolog()

with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.log_metric("test_accuracy", accuracy)
    joblib.dump(model, "models/model.joblib")
```

Register the best model in the MLflow Model Registry with an alias (e.g., `champion`).

### 6. Model serving with FastAPI

Create `src/api/config.yaml`:

```yaml
model_path: "models/model.joblib"
port: 8000
```

Create `src/api/app.py` — load the model at startup via a lifespan handler, expose `/health` and `/predict` endpoints. Use Pydantic models for request/response validation.

```bash
uv add fastapi "uvicorn[standard]" pyyaml
uv run uvicorn src.api.app:app --reload
```

Test:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[...]]}'
```

### 7. Frontend with Streamlit

Create `src/frontend/config.yaml`:

```yaml
api_url: "http://localhost:8000"
```

Build a Streamlit form matching the model's input features. Use `st.selectbox` for categorical fields, `st.number_input` for numerical fields. POST to the API on form submission and display the result.

```bash
uv add streamlit requests
uv run streamlit run src/frontend/app.py
```

### 8. Testing

Write tests in `tests/` using pytest. Use `tmp_path` for file-based tests:

```python
def test_load_data(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n")
    df = load_data(csv)
    assert len(df) == 1
```

```bash
uv run pytest
```

## Key Principles

- **Reproducibility**: code (Git) + data (DVC) + environment (uv + `uv.lock`) + experiments (MLflow)
- **Single artifact**: embed preprocessing in the sklearn `Pipeline` so the saved model includes transforms
- **Configuration over code**: use YAML config files for paths, ports, and URLs — not hardcoded values
- **Quality gates**: pre-commit hooks (ruff + mypy) run on every commit
- **Separation of concerns**: training code, API serving, and frontend live in distinct directories with their own configs

## Common Commands

```bash
uv sync                                        # Install/update all dependencies
uv run pytest                                  # Run tests
uv run ruff check src/ tests/                  # Lint
uv run ruff format src/ tests/                 # Format
uv run pre-commit run --all-files              # Run all quality hooks
uv run mlflow server --host 127.0.0.1 --port 8080  # Start MLflow
uv run python -m <package>.train               # Train model
uv run uvicorn src.api.app:app --reload        # Start API
uv run streamlit run src/frontend/app.py       # Start frontend
dvc repro                                      # Run DVC pipeline
dvc push                                       # Push data to remote
```
