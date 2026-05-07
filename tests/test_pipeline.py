"""Smoke tests for the Credit Card Default Risk Analysis project."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from load_data import load_credit_card_data
from train_model import train


def test_sample_data_loads() -> None:
    sample_path = PROJECT_ROOT / "data" / "sample" / "credit_card_default_sample.csv"
    df = load_credit_card_data(sample_path)
    assert len(df) > 0
    assert "default_next_month" in df.columns
    assert df["default_next_month"].isin([0, 1]).all()


def test_training_pipeline_runs(tmp_path) -> None:
    sample_path = PROJECT_ROOT / "data" / "sample" / "credit_card_default_sample.csv"
    summary = train(sample_path, output_dir=tmp_path / "models", reports_dir=tmp_path / "reports")
    assert summary["best_model"] in {"logistic_regression", "random_forest"}
    assert Path(summary["model_path"]).exists()
    assert 0 <= summary["metrics"]["roc_auc"] <= 1
