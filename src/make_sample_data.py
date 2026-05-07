"""Generate a synthetic UCI-compatible sample dataset.

The included sample data is NOT real banking data. It exists so the repository
can run immediately before a user downloads the full public UCI/Kaggle dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import SAMPLE_DATA_DIR
except ImportError:  # pragma: no cover
    from src.config import SAMPLE_DATA_DIR


def make_sample_data(rows: int = 1500, output_path: str | Path | None = None) -> Path:
    """Create a realistic synthetic sample with UCI-compatible column names."""
    rng = np.random.default_rng(42)
    output = Path(output_path) if output_path else SAMPLE_DATA_DIR / "credit_card_default_sample.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    limit_bal = rng.choice([20000, 50000, 80000, 120000, 200000, 300000, 500000], size=rows)
    sex = rng.choice([1, 2], size=rows, p=[0.42, 0.58])
    education = rng.choice([1, 2, 3, 4], size=rows, p=[0.25, 0.45, 0.22, 0.08])
    marriage = rng.choice([1, 2, 3], size=rows, p=[0.44, 0.49, 0.07])
    age = rng.integers(21, 68, size=rows)

    repayment_base = rng.choice([-1, 0, 1, 2, 3], size=rows, p=[0.22, 0.50, 0.14, 0.10, 0.04])
    pay_cols: dict[str, np.ndarray] = {}
    for month in [0, 2, 3, 4, 5, 6]:
        noise = rng.choice([-1, 0, 1], size=rows, p=[0.18, 0.64, 0.18])
        pay_cols[f"PAY_{month}"] = np.clip(repayment_base + noise, -1, 8)

    utilization = rng.uniform(0.05, 1.15, size=rows)
    bill_cols: dict[str, np.ndarray] = {}
    payment_cols: dict[str, np.ndarray] = {}
    for month in range(1, 7):
        bill_amount = (limit_bal * utilization * rng.uniform(0.70, 1.05, size=rows)).astype(int)
        bill_cols[f"BILL_AMT{month}"] = bill_amount
        payment_ratio = rng.uniform(0.02, 0.35, size=rows)
        payment_cols[f"PAY_AMT{month}"] = np.maximum(0, (bill_amount * payment_ratio).astype(int))

    late_payment_score = sum(pay_cols[column] for column in pay_cols) / 6
    utilization_score = np.mean([bill_cols[f"BILL_AMT{i}"] / limit_bal for i in range(1, 7)], axis=0)
    payment_coverage_score = np.mean(
        [payment_cols[f"PAY_AMT{i}"] / np.maximum(bill_cols[f"BILL_AMT{i}"], 1) for i in range(1, 7)],
        axis=0,
    )
    low_payment_score = 1 - payment_coverage_score

    logit = -2.3 + 0.78 * late_payment_score + 1.25 * utilization_score + 0.90 * low_payment_score
    default_probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, np.clip(default_probability, 0.02, 0.92))

    df = pd.DataFrame(
        {
            "ID": np.arange(1, rows + 1),
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **pay_cols,
            **bill_cols,
            **payment_cols,
            "default.payment.next.month": target,
        }
    )
    df.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic sample credit-card default dataset.")
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--output", default=SAMPLE_DATA_DIR / "credit_card_default_sample.csv")
    args = parser.parse_args()
    print(f"Sample dataset saved to: {make_sample_data(args.rows, args.output)}")


if __name__ == "__main__":
    main()
