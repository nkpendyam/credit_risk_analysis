"""Train credit-card default risk models and save evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from config import FEATURE_COLUMNS, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN, TEST_SIZE
    from load_data import load_credit_card_data
    from preprocess import build_preprocessor
except ImportError:  # pragma: no cover
    from src.config import FEATURE_COLUMNS, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN, TEST_SIZE
    from src.load_data import load_credit_card_data
    from src.preprocess import build_preprocessor


def build_models() -> dict[str, Pipeline]:
    """Return candidate models wrapped in preprocessing pipelines."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=180,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def get_positive_class_scores(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities or normalized decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        denominator = scores.max() - scores.min()
        if denominator == 0:
            return np.zeros_like(scores, dtype=float)
        return (scores - scores.min()) / denominator
    return model.predict(X_test)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a classifier using credit-risk-friendly metrics."""
    y_pred = model.predict(X_test)
    y_score = get_positive_class_scores(model, X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }


def create_risk_segments(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Create customer-level predictions with risk-band labels."""
    scores = get_positive_class_scores(model, X_test)
    report = X_test.copy()
    report["actual_default"] = y_test.to_numpy()
    report["predicted_default_probability"] = scores
    report["risk_band"] = pd.cut(
        report["predicted_default_probability"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Very High"],
    )
    return report


def summarize_risk_segments(segment_report: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed and predicted default risk by risk band."""
    return (
        segment_report.groupby("risk_band", observed=True)
        .agg(
            customers=("actual_default", "size"),
            observed_default_rate=("actual_default", "mean"),
            avg_predicted_default_probability=("predicted_default_probability", "mean"),
        )
        .reset_index()
    )


def train(
    data_path: str | Path,
    output_dir: str | Path = MODELS_DIR,
    reports_dir: str | Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Train candidate models and save the best model by ROC-AUC."""
    output_path = Path(output_dir)
    report_path = Path(reports_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    df = load_credit_card_data(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results: dict[str, Any] = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        results[model_name] = evaluate_model(model, X_test, y_test)

    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = models[best_model_name]

    model_file = output_path / "best_credit_risk_model.joblib"
    joblib.dump(best_model, model_file)

    metrics_rows = [
        {
            "model": model_name,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        for model_name, metrics in results.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(report_path / "model_metrics.csv", index=False)

    segment_report = create_risk_segments(best_model, X_test, y_test)
    segment_report.to_csv(report_path / "risk_scored_customers.csv", index=False)
    summarize_risk_segments(segment_report).to_csv(report_path / "risk_segment_summary.csv", index=False)

    with open(report_path / "model_metrics_full.json", "w", encoding="utf-8") as file:
        json.dump({"best_model": best_model_name, "results": results}, file, indent=2)

    return {
        "rows": int(len(df)),
        "default_rate": float(y.mean()),
        "best_model": best_model_name,
        "model_path": str(model_file),
        "reports_dir": str(report_path),
        "metrics": results[best_model_name],
    }


def main() -> None:
    default_sample = SAMPLE_DATA_DIR / "credit_card_default_sample.csv"
    parser = argparse.ArgumentParser(description="Train credit-card default risk models.")
    parser.add_argument("--data-path", default=default_sample, help="Path to CSV/XLS/XLSX dataset.")
    parser.add_argument("--output-dir", default=MODELS_DIR)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    args = parser.parse_args()

    summary = train(args.data_path, args.output_dir, args.reports_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
