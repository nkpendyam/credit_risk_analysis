"""Reusable preprocessing pipeline for credit-card default models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
except ImportError:  # pragma: no cover
    from src.config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing transformer for categorical and numeric fields."""
    numeric_columns = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )
