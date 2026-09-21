# Original-source verification

Local verification completed 21 September 2026.

- Original UCI CSV: 30,000 records, 6,636 positive targets; source hash and attribution recorded in `model_metrics_full.json`.
- Identical feature groups stay together. Train/validation/test sizes: 17,999 / 6,001 / 6,000; pairwise feature-group overlap is zero.
- Validation ROC AUC selects random forest. Only that train-fitted model and a prior baseline are evaluated on test data. Test ROC AUC: 0.78317125; average precision: 0.55722406; baseline ROC AUC: 0.5.
- Twelve pytest tests passed. Cases cover training, malformed targets, nonfinite features, deterministic split isolation, unlabelled CSV/XLSX prediction, and Excel header detection.
- The report generator ran successfully. Its two PNG charts were visually inspected for readable labels and clipping. The self-contained HTML was generated; no browser runtime check is claimed.
- `git diff --check` passes. Raw data, serialized models and row-level prediction files are excluded from Git; aggregate evidence remains included.

These are retrospective model scores, not calibrated lending probabilities. No fairness, deployment readiness, causal effect or financial saving is established.
