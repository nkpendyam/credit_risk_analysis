# Dataset Notes

This repository is built for the public **Default of Credit Card Clients** dataset.

## Recommended dataset

- Source: UCI Machine Learning Repository
- Dataset: Default of Credit Card Clients
- Task: Binary classification
- Target: Whether a client defaults next month
- Rows in full dataset: 30,000
- Main features: credit limit, repayment status, bill amount, payment amount, age, education, sex, and marital status
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Kaggle mirror

Kaggle also hosts a mirror under:

```text
uciml/default-of-credit-card-clients-dataset
```

The Kaggle API requires account authentication, so the full Kaggle dataset is not committed to this repository.

## Included sample data

`data/sample/credit_card_default_sample.csv` is a synthetic sample with the same column format as the UCI/Kaggle dataset. It is included only so the code, tests, and README visuals can run immediately.

Do not present the sample data as real bank customer data.
