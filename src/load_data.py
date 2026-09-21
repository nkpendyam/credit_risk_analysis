"""Data loading, standardization, and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from config import FEATURE_COLUMNS, TARGET_COLUMN
except ImportError:  # pragma: no cover - supports package-style imports in IDEs
    from src.config import FEATURE_COLUMNS, TARGET_COLUMN


TARGET_ALIASES = {
    "default.payment.next.month": TARGET_COLUMN,
    "default payment next month": TARGET_COLUMN,
    "default_next_month": TARGET_COLUMN,
    "Y": TARGET_COLUMN,
    "y": TARGET_COLUMN,
    "target": TARGET_COLUMN,
}

UCI_COLUMN_MAP = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}


def read_credit_card_file(path: str | Path) -> pd.DataFrame:
    """Read a CSV/XLS/XLSX file into a dataframe.

    The original UCI Excel file sometimes contains a title row before the real
    header. This function handles both the Kaggle CSV and UCI Excel formats.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xls", ".xlsx"}:
        first_try = pd.read_excel(file_path)
        standardized_columns = set(standardize_columns(first_try).columns)
        if set(FEATURE_COLUMNS).issubset(standardized_columns):
            return first_try
        return pd.read_excel(file_path, header=1)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV, XLS, or XLSX.")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert UCI/Kaggle column names into consistent project names."""
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    rename_map = {**UCI_COLUMN_MAP, **TARGET_ALIASES}
    return cleaned.rename(columns=rename_map)


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a readable error if required model columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Use the UCI/Kaggle credit-card default dataset format."
        )


def clean_credit_card_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean model-ready dataframe.

    Decisions:
    - Drop ID because it is an identifier, not a model feature.
    - Convert all required fields to numeric values.
    - Map unknown EDUCATION categories 0/5/6 to 4 = others/unknown.
    - Map unknown MARRIAGE category 0 to 3 = others.
    - Reject non-parsable or non-finite required values so row counts stay
      auditable rather than changing silently.
    - Validate the binary target before converting it to integer values.
    """
    cleaned = standardize_columns(df)
    if "ID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["ID"])

    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    validate_columns(cleaned, required_columns)

    cleaned = cleaned[required_columns].copy()
    numeric = cleaned.apply(pd.to_numeric, errors="coerce")
    invalid_features = [
        column
        for column in FEATURE_COLUMNS
        if numeric[column].isna().any() or not np.isfinite(numeric[column]).all()
    ]
    if invalid_features:
        raise ValueError(
            "Required features contain missing, non-numeric, or non-finite values: "
            + ", ".join(invalid_features)
        )

    target = numeric[TARGET_COLUMN]
    if target.isna().any() or not np.isfinite(target).all() or not target.isin([0, 1]).all():
        raise ValueError(
            f"{TARGET_COLUMN} must contain only finite binary values 0 and 1 before integer conversion."
        )
    cleaned = numeric

    cleaned["EDUCATION"] = cleaned["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    cleaned["MARRIAGE"] = cleaned["MARRIAGE"].replace({0: 3})
    cleaned[TARGET_COLUMN] = target.astype(int)

    return cleaned


def load_credit_card_data(path: str | Path) -> pd.DataFrame:
    """Load and clean a credit-card default dataset file."""
    return clean_credit_card_data(read_credit_card_file(path))
