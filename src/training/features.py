import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_features_pipeline(categorical_features: list[str], numerical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
            ("num", "passthrough", numerical_features),
        ],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
    ])


def build_features(df: pd.DataFrame, categorical_features: list[str], numerical_features: list[str]) -> pd.DataFrame:
    pipeline = build_features_pipeline(categorical_features, numerical_features)
    df_transformed = pipeline.fit_transform(df)
    return df_transformed
