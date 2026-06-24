from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# def build_pipeline(categorical_features: list[str]) -> Pipeline:
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
#         ],
#         remainder="passthrough",
#     )
#     return Pipeline([
#         ("preprocessor", preprocessor),
#         ("classifier", RandomForestClassifier(random_state=42)),
#     ])

from training.features import build_features_pipeline

def build_pipeline(categorical_features: list[str], numerical_features: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_features_pipeline(categorical_features, numerical_features)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])


def train(data_path: str | Path) -> tuple[GridSearchCV, float]:
    df = pd.read_csv(data_path)

    target = "LeaveOrNot"
    X = df.drop(columns=[target])
    y = df[target]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(categorical_features, numerical_features)

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10],
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    test_accuracy = grid_search.score(X_test, y_test)
    
    return grid_search, test_accuracy


if __name__ == "__main__":
    grid_search, test_accuracy = train(Path(__file__).resolve().parents[2] / "data" / "employees.csv")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    print(f"Test accuracy:    {test_accuracy:.4f}")
    
