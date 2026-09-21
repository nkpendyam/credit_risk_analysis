"""Build a static, evidence-only model validation report."""
from __future__ import annotations

import base64
import csv
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "uci"


def chart_bytes(plotter, path, alt):
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    plotter(ax)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_report():
    metrics_path = OUT / "model_metrics_full.json"
    segments_path = OUT / "risk_segment_summary.csv"
    if not metrics_path.exists() or not segments_path.exists():
        raise FileNotFoundError("Expected model_metrics_full.json and risk_segment_summary.csv in reports/uci")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with segments_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selection = metrics.get("selection")
    test_block = metrics.get("test")
    if not isinstance(selection, dict) or not isinstance(test_block, dict):
        raise ValueError("model metrics must contain selection and test objects")
    validation = selection.get("validation_candidates")
    test = test_block.get("selected_model_metrics")
    baseline = test_block.get("dummy_prior_baseline_metrics")
    if not isinstance(validation, dict) or not isinstance(test, dict) or not isinstance(baseline, dict):
        raise ValueError("selection.validation_candidates and test metric objects are required")
    pairs = []
    for name, value in validation.items():
        if not isinstance(value, dict) or not isinstance(value.get("roc_auc"), (int, float)):
            raise ValueError(f"validation_results.{name}.roc_auc is required")
        pairs.append((str(name), float(value["roc_auc"])))
    if not isinstance(test.get("roc_auc"), (int, float)) or not isinstance(baseline.get("roc_auc"), (int, float)):
        raise ValueError("test_results.roc_auc is required")
    pairs.append(("final held-out test", float(test["roc_auc"])))
    pairs.append(("dummy prior baseline", float(baseline["roc_auc"])))
    required = {"risk_band", "customers", "observed_default_rate", "avg_model_score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"risk segment CSV must contain {sorted(required)}")
    def auc_plot(ax):
        labels = [x[0] for x in pairs]; values = [x[1] for x in pairs]
        ax.bar(labels, values, color="#2563eb")
        ax.set_ylim(0, 1); ax.set_ylabel("ROC AUC"); ax.set_title("Validation candidate and held-out test ROC AUC")
        ax.tick_params(axis="x", rotation=25)
    def segment_plot(ax):
        names = [r["risk_band"] for r in rows]
        counts = [float(r["customers"]) for r in rows]
        rates = [float(r["observed_default_rate"]) for r in rows]
        ax2 = ax.twinx(); ax.bar(names, counts, color="#cbd5e1", label="records"); ax2.plot(names, rates, "o-", color="#dc2626", label="observed rate")
        ax.set_ylabel("Records"); ax2.set_ylabel("Observed rate"); ax.set_title("Risk score bands: volume and observed outcome rate")
        ax.tick_params(axis="x", rotation=25)
    OUT.mkdir(parents=True, exist_ok=True)
    auc = chart_bytes(auc_plot, OUT / "roc_auc_comparison.png", "ROC AUC comparison chart")
    segment = chart_bytes(segment_plot, OUT / "risk_segment_summary.png", "Risk band counts and observed default rates chart")
    html = f'''<!doctype html><meta charset="utf-8"><title>Credit default evidence report</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;color:#172033}}img{{max-width:100%}}.note{{background:#f1f5f9;padding:1rem}}</style>
<h1>Credit default model evidence</h1><p>Retrospective validation of the Taiwan 2005 credit-card default cohort.</p>
<h2>ROC AUC comparison</h2><img alt="ROC AUC comparison chart" src="data:image/png;base64,{auc}">
<h2>Risk score bands</h2><img alt="Risk band counts and observed default rates chart" src="data:image/png;base64,{segment}">
<div class="note"><h2>Interpretation and limits</h2><p>Scores and observed rates describe this historical cohort. They are not calibrated probabilities, lending recommendations, or evidence of business savings. Risk bands show retrospective counts and outcomes, not causal treatment effects.</p><p>Source: <a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients">UCI Default of Credit Card Clients</a>, DOI 10.24432/C55S3H, CC BY 4.0.</p></div>'''
    (OUT / "analysis.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    make_report()
