"""Train and honestly evaluate retrospective credit-default models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

try:
    from config import FEATURE_COLUMNS, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN
    from load_data import load_credit_card_data
    from preprocess import build_preprocessor
except ImportError:  # pragma: no cover
    from src.config import FEATURE_COLUMNS, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN
    from src.load_data import load_credit_card_data
    from src.preprocess import build_preprocessor


UCI_SOURCE = {
    "uci_dataset_id": 350,
    "doi": "10.24432/C55S3H",
    "license": "CC BY 4.0",
    "cohort": "Taiwan credit-card clients, 2005 retrospective cohort",
}
UCI_RAW_SHA256 = "20a2540061fab3984f208fbe55745b0d60e3a86207c8b02e408fd23f1a3a4b94"


def build_models() -> dict[str, Pipeline]:
    """Return candidate pipelines. They are selected solely by validation ROC AUC."""
    return {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )),
        ]),
    }


def get_positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return model scores for the positive class without interpreting them as calibrated decisions."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        span = scores.max() - scores.min()
        return np.zeros_like(scores, dtype=float) if span == 0 else (scores - scores.min()) / span
    return model.predict(features)


def evaluate_model(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Calculate descriptive discrimination and classification metrics for one split."""
    predicted = model.predict(features)
    scores = get_positive_class_scores(model, features)
    return {
        "roc_auc": float(roc_auc_score(target, scores)),
        "pr_auc": float(average_precision_score(target, scores)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, predicted).tolist(),
        "classification_report": classification_report(target, predicted, zero_division=0, output_dict=True),
    }


def _feature_groups(features: pd.DataFrame) -> np.ndarray:
    """Give identical feature vectors the same group, preventing feature leakage across splits."""
    return pd.factorize(pd.MultiIndex.from_frame(features), sort=False)[0]


def _split_indices(features: pd.DataFrame, target: pd.Series) -> dict[str, np.ndarray]:
    """Create reproducible, group-aware approximate 60/20/20 stratified splits."""
    groups = _feature_groups(features)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_validation_idx, test_idx = next(outer.split(features, target, groups))

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
    inner_train_idx, inner_validation_idx = next(inner.split(
        features.iloc[train_validation_idx], target.iloc[train_validation_idx], groups[train_validation_idx]
    ))
    return {
        "train": train_validation_idx[inner_train_idx],
        "validation": train_validation_idx[inner_validation_idx],
        "test": test_idx,
    }


def _indices_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(",".join(map(str, sorted(map(int, indices)))).encode("utf-8")).hexdigest()


def _split_manifest(indices: dict[str, np.ndarray], groups: np.ndarray) -> dict[str, Any]:
    group_sets = {name: set(map(int, groups[values])) for name, values in indices.items()}
    return {
        name: {
            "rows": int(len(values)),
            "row_indices_sha256": _indices_hash(values),
            "feature_group_ids_sha256": _indices_hash(np.array(sorted(group_sets[name]))),
        }
        for name, values in indices.items()
    } | {
        "feature_group_overlap": {
            "train_validation": int(len(group_sets["train"] & group_sets["validation"])),
            "train_test": int(len(group_sets["train"] & group_sets["test"])),
            "validation_test": int(len(group_sets["validation"] & group_sets["test"])),
        }
    }


def create_risk_segments(model: Any, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Create descriptive score bands for held-out records; these are not credit decisions."""
    scores = get_positive_class_scores(model, features)
    report = features.copy()
    report["actual_default"] = target.to_numpy()
    report["model_score"] = scores
    report["risk_band"] = pd.cut(
        report["model_score"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Score 0.00-0.25", "Score 0.25-0.50", "Score 0.50-0.75", "Score 0.75-1.00"],
    )
    return report


def summarize_risk_segments(segment_report: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed outcomes by descriptive held-out score bands."""
    summary = segment_report.groupby("risk_band", observed=True).agg(
        customers=("actual_default", "size"),
        observed_default_rate=("actual_default", "mean"),
        avg_model_score=("model_score", "mean"),
    ).reset_index()
    # Compatibility aliases make the evidence renderer explicit about these bands.
    summary["risk_segment"] = summary["risk_band"].astype(str)
    summary["records"] = summary["customers"]
    summary["observed_rate"] = summary["observed_default_rate"]
    return summary


def _metrics_rows(validation_results: dict[str, dict[str, Any]], selected_name: str, selected_test: dict[str, Any], dummy_test: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"model": name, "split": "validation", "selected": name == selected_name,
         **{metric: result[metric] for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall")}}
        for name, result in validation_results.items()
    ]
    rows.extend([
        {"model": selected_name, "split": "test", "selected": True,
         **{metric: selected_test[metric] for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall")}},
        {"model": "dummy_prior_baseline", "split": "test", "selected": False,
         **{metric: dummy_test[metric] for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall")}},
    ])
    return rows


def _source_manifest(data_path: Path, df: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Attribute the UCI cohort only when the exact verified raw source is used."""
    source_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    profile = {
        "path": str(data_path),
        "sha256": source_hash,
        "raw_rows": int(len(df)),
        "positive_targets": int(target.sum()),
        "exact_duplicate_rows_without_identifier": int(df.duplicated().sum()),
    }
    if source_hash == UCI_RAW_SHA256:
        return {"source_kind": "uci_original", **profile, **UCI_SOURCE}
    return {"source_kind": "custom_or_synthetic", **profile}


def train(data_path: str | Path, output_dir: str | Path = MODELS_DIR, reports_dir: str | Path = REPORTS_DIR) -> dict[str, Any]:
    """Select candidates on validation data, then evaluate only the selected model on test data."""
    data_path = Path(data_path)
    output_path = Path(output_dir)
    report_path = Path(reports_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    df = load_credit_card_data(data_path)
    features = df[FEATURE_COLUMNS]
    target = df[TARGET_COLUMN]
    indices = _split_indices(features, target)
    groups = _feature_groups(features)
    if any(value != 0 for value in _split_manifest(indices, groups)["feature_group_overlap"].values()):
        raise AssertionError("identical feature vectors crossed a data split")

    train_features, train_target = features.iloc[indices["train"]], target.iloc[indices["train"]]
    validation_features, validation_target = features.iloc[indices["validation"]], target.iloc[indices["validation"]]
    test_features, test_target = features.iloc[indices["test"]], target.iloc[indices["test"]]

    models = build_models()
    validation_results: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        model.fit(train_features, train_target)
        validation_results[model_name] = evaluate_model(model, validation_features, validation_target)
    selected_name = max(validation_results, key=lambda name: validation_results[name]["roc_auc"])
    selected_model = models[selected_name]  # Retain the train-only fitted model; never refit after selection.

    selected_test = evaluate_model(selected_model, test_features, test_target)
    dummy_model = DummyClassifier(strategy="prior")
    dummy_model.fit(train_features, train_target)
    dummy_test = evaluate_model(dummy_model, test_features, test_target)

    model_file = output_path / "best_credit_risk_model.joblib"
    joblib.dump(selected_model, model_file)
    pd.DataFrame(_metrics_rows(validation_results, selected_name, selected_test, dummy_test)).to_csv(report_path / "model_metrics.csv", index=False)

    segment_report = create_risk_segments(selected_model, test_features, test_target)
    segment_report.to_csv(report_path / "risk_scored_customers.csv", index=False)
    summarize_risk_segments(segment_report).to_csv(report_path / "risk_segment_summary.csv", index=False)

    manifest = {
        "source": _source_manifest(data_path, df, target),
        "splits": _split_manifest(indices, groups),
        "selection": {"metric": "validation_roc_auc", "selected_model": selected_name, "validation_candidates": validation_results},
        "test": {"selected_model_metrics": selected_test, "dummy_prior_baseline_metrics": dummy_test},
        # Flat metric entries are consumed by the static evidence-report renderer.
        **{f"validation_{name}": metrics for name, metrics in validation_results.items()},
        "selected_test": selected_test,
        "dummy_prior_baseline_test": dummy_test,
        "limitations": [
            "Candidate selection used validation ROC AUC; the held-out test split was evaluated only for the selected model and dummy baseline.",
            "Feature-identical records are group-isolated across splits; split sizes are approximate because whole groups are retained.",
            "Score bands are descriptive retrospective model-score bands, not calibrated probabilities, credit decisions, or lending recommendations.",
        ],
    }
    (report_path / "model_metrics_full.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "rows": int(len(df)), "default_rate": float(target.mean()), "best_model": selected_name,
        "model_path": str(model_file), "reports_dir": str(report_path), "metrics": selected_test,
        "validation_metrics": validation_results, "dummy_baseline_test_metrics": dummy_test,
        "split_sizes": {name: int(len(values)) for name, values in indices.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train retrospective credit-default models with held-out evaluation.")
    parser.add_argument("--data-path", default=SAMPLE_DATA_DIR / "credit_card_default_sample.csv", help="Path to CSV/XLS/XLSX dataset.")
    parser.add_argument("--output-dir", default=MODELS_DIR)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.data_path, args.output_dir, args.reports_dir), indent=2))


if __name__ == "__main__":
    main()
