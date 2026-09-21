"""Generate credit-card default risk predictions from a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from config import MODELS_DIR, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN
    from load_data import clean_credit_card_data, read_credit_card_file, standardize_columns
except ImportError:  # pragma: no cover
    from src.config import MODELS_DIR, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN
    from src.load_data import clean_credit_card_data, read_credit_card_file, standardize_columns


def predict(model_path: str | Path, input_path: str | Path, output_path: str | Path) -> Path:
    """Load a trained model and write customer-level risk predictions."""
    model = joblib.load(model_path)
    raw = standardize_columns(read_credit_card_file(input_path))
    if TARGET_COLUMN not in raw.columns:
        raw[TARGET_COLUMN] = 0  # Validation placeholder; excluded from model features and output.
    df = clean_credit_card_data(raw)
    X = df.drop(columns=[TARGET_COLUMN])

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    result = X.copy()
    result["predicted_default"] = predictions
    result["model_score"] = probabilities
    result["risk_band"] = pd.cut(
        result["model_score"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Score 0.00-0.25", "Score 0.25-0.50", "Score 0.50-0.75", "Score 0.75-1.00"],
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict default risk for credit-card customers.")
    parser.add_argument("--model-path", default=MODELS_DIR / "best_credit_risk_model.joblib")
    parser.add_argument("--input", default=SAMPLE_DATA_DIR / "credit_card_default_sample.csv")
    parser.add_argument("--output", default=REPORTS_DIR / "predictions.csv")
    args = parser.parse_args()

    output_path = predict(args.model_path, args.input, args.output)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
