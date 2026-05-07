"""Download the public credit-card default dataset.

Supported sources:
1. UCI Machine Learning Repository through `ucimlrepo`.
2. Kaggle mirror through the Kaggle API.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from config import RAW_DATA_DIR
except ImportError:  # pragma: no cover
    from src.config import RAW_DATA_DIR


def download_from_uci(output_dir: str | Path = RAW_DATA_DIR) -> Path:
    """Download the UCI dataset and save it as a CSV file."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install ucimlrepo first: pip install ucimlrepo") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features
    target = dataset.data.targets
    output_path = output / "UCI_Credit_Card.csv"
    features.join(target).to_csv(output_path, index=False)
    return output_path


def download_from_kaggle(output_dir: str | Path = RAW_DATA_DIR) -> Path:
    """Download the Kaggle mirror of the UCI dataset.

    Requires Kaggle credentials. See the README for setup instructions.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "uciml/default-of-credit-card-clients-dataset",
        "-p",
        str(output),
        "--unzip",
    ]
    subprocess.run(command, check=True)
    expected = output / "UCI_Credit_Card.csv"
    if not expected.exists():
        raise FileNotFoundError(f"Kaggle download completed, but expected file was not found: {expected}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the credit-card default dataset.")
    parser.add_argument("--source", choices=["uci", "kaggle"], default="uci")
    parser.add_argument("--output-dir", default=RAW_DATA_DIR)
    args = parser.parse_args()

    if args.source == "uci":
        output_path = download_from_uci(args.output_dir)
    else:
        output_path = download_from_kaggle(args.output_dir)
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
