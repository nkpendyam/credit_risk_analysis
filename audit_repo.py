"""Repository audit script.

Run this before pushing to GitHub:
    python audit_repo.py
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "LICENSE",
    ".gitignore",
    "data/README.md",
    "data/sample/credit_card_default_sample.csv",
    "src/config.py",
    "src/load_data.py",
    "src/preprocess.py",
    "src/train_model.py",
    "src/predict.py",
    "src/make_report_figures.py",
    "src/download_data.py",
    "tests/test_pipeline.py",
    "reports/model_metrics.csv",
    "reports/risk_segment_summary.csv",
    "reports/figures/01_default_distribution.png",
    "reports/figures/02_default_by_payment_status.png",
    "reports/figures/03_model_metrics.png",
    "reports/figures/04_risk_segments.png",
    "reports/figures/05_confusion_matrix.png",
]


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"- {path}")
        return 1

    for python_file in SRC.glob("*.py"):
        py_compile.compile(str(python_file), doraise=True)

    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True)
    if result.returncode != 0:
        return result.returncode

    print("Audit passed: repo files exist, Python files compile, tests pass, and README figures exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
