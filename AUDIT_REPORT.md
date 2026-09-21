# Credit-risk project audit

## Scope and corrected authority

The project has been reframed around the original UCI **Default of Credit Card Clients** dataset. The authoritative study uses `data/raw/UCI_Credit_Card.csv`, writes model outputs to `models/uci/`, and writes evidence artifacts to `reports/uci/`. Existing root `reports/` files remain explicitly labeled historical synthetic-demo artifacts; they are not used for UCI conclusions.

## Reproduction contract

```text
python src/download_data.py --source uci
python src/train_model.py --data-path data/raw/UCI_Credit_Card.csv --reports-dir reports/uci --output-dir models/uci
python src/build_evidence_report.py
pytest -q
```

The evaluation design is a 60/20/20 train/validation/test split. Validation selects the model and any comparison settings. Only the selected model and a dummy baseline are evaluated on the held-out test partition. The report generator builds figures from the resulting UCI output files; it does not convert historical synthetic-demo numbers into current results.

## Interpretation boundary

This is a historical, observational study of Taiwanese credit-card clients in 2005. It can support exploratory monitoring and model-validation discussion. It cannot establish causal effects, current portfolio performance, calibrated probabilities, fair treatment, or lending/collections decisions. Score bands are descriptive ranking aids. Protected demographic fields are present in the legacy source, and no fairness assurance is implied.

## Audit status

The original-source run completed on 30,000 records. Validation selected random forest (ROC AUC 0.7834); its held-out test ROC AUC is 0.7832 and average precision is 0.5572. Exact feature groups have zero overlap between the 17,999/6,001/6,000 train/validation/test partitions. The old test-based model selection was removed. Preprocessing and both candidates fit on training data only; no post-selection refit uses validation or test data.

Invalid targets and nonfinite features now raise errors instead of silently truncating targets or dropping records. Prediction exports use descriptive model scores and accept unlabelled input. Source attribution is tied to the verified original-file fingerprint. Unit tests, generated charts and detailed manifests accompany the implementation; final test status is recorded in `reports/uci/VERIFICATION.md`.

Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), DOI `10.24432/C55S3H`, CC BY 4.0. Citation: Yeh and Lien (2009).
