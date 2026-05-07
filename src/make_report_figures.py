"""Create professional README/report figures for the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from config import FIGURES_DIR, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN
    from load_data import load_credit_card_data
except ImportError:  # pragma: no cover
    from src.config import FIGURES_DIR, REPORTS_DIR, SAMPLE_DATA_DIR, TARGET_COLUMN
    from src.load_data import load_credit_card_data


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_default_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No Default", "Default"]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, counts.values)
    plt.title("Default Distribution")
    plt.ylabel("Customers")
    for index, value in enumerate(counts.values):
        plt.text(index, value, str(value), ha="center", va="bottom")
    _save_current_figure(output_path)


def plot_default_by_payment_status(df: pd.DataFrame, output_path: Path) -> None:
    grouped = df.groupby("PAY_0")[TARGET_COLUMN].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(grouped["PAY_0"], grouped[TARGET_COLUMN], marker="o")
    plt.title("Observed Default Rate by Latest Repayment Status")
    plt.xlabel("PAY_0 Repayment Status")
    plt.ylabel("Observed Default Rate")
    plt.ylim(0, max(0.05, grouped[TARGET_COLUMN].max() * 1.15))
    _save_current_figure(output_path)


def plot_model_metrics(metrics_path: Path, output_path: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    display_metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    melted = metrics.melt(id_vars="model", value_vars=display_metrics, var_name="metric", value_name="score")

    plt.figure(figsize=(9, 4.8))
    models = list(metrics["model"])
    x_positions = range(len(display_metrics))
    width = 0.35
    for offset, model_name in enumerate(models):
        model_scores = [
            melted[(melted["model"] == model_name) & (melted["metric"] == metric)]["score"].iloc[0]
            for metric in display_metrics
        ]
        shifted_positions = [position + (offset - 0.5) * width for position in x_positions]
        plt.bar(shifted_positions, model_scores, width=width, label=model_name.replace("_", " ").title())
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(list(x_positions), [metric.upper().replace("_", "-") for metric in display_metrics])
    plt.legend()
    _save_current_figure(output_path)


def plot_risk_segments(summary_path: Path, output_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    plt.figure(figsize=(8, 4.5))
    plt.bar(summary["risk_band"].astype(str), summary["observed_default_rate"])
    plt.title("Observed Default Rate by Risk Band")
    plt.xlabel("Risk Band")
    plt.ylabel("Observed Default Rate")
    plt.ylim(0, max(0.05, summary["observed_default_rate"].max() * 1.20))
    for index, value in enumerate(summary["observed_default_rate"]):
        plt.text(index, value, f"{value:.1%}", ha="center", va="bottom")
    _save_current_figure(output_path)


def plot_confusion_matrix(metrics_full_path: Path, output_path: Path) -> None:
    with open(metrics_full_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    best_model = payload["best_model"]
    matrix = payload["results"][best_model]["confusion_matrix"]

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(matrix)
    plt.title(f"Confusion Matrix - {best_model.replace('_', ' ').title()}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["No Default", "Default"])
    plt.yticks([0, 1], ["No Default", "Default"])
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            plt.text(col_index, row_index, str(value), ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
    _save_current_figure(output_path)


def make_figures(
    data_path: str | Path = SAMPLE_DATA_DIR / "credit_card_default_sample.csv",
    reports_dir: str | Path = REPORTS_DIR,
    figures_dir: str | Path = FIGURES_DIR,
) -> list[Path]:
    """Create all figures and return their paths."""
    data_file = Path(data_path)
    report_dir = Path(reports_dir)
    figure_dir = Path(figures_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = load_credit_card_data(data_file)
    output_paths = [
        figure_dir / "01_default_distribution.png",
        figure_dir / "02_default_by_payment_status.png",
        figure_dir / "03_model_metrics.png",
        figure_dir / "04_risk_segments.png",
        figure_dir / "05_confusion_matrix.png",
    ]

    plot_default_distribution(df, output_paths[0])
    plot_default_by_payment_status(df, output_paths[1])
    plot_model_metrics(report_dir / "model_metrics.csv", output_paths[2])
    plot_risk_segments(report_dir / "risk_segment_summary.csv", output_paths[3])
    plot_confusion_matrix(report_dir / "model_metrics_full.json", output_paths[4])
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create README/report figures.")
    parser.add_argument("--data-path", default=SAMPLE_DATA_DIR / "credit_card_default_sample.csv")
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    parser.add_argument("--figures-dir", default=FIGURES_DIR)
    args = parser.parse_args()

    for path in make_figures(args.data_path, args.reports_dir, args.figures_dir):
        print(f"Created figure: {path}")


if __name__ == "__main__":
    main()
