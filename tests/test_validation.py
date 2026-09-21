import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.load_data import clean_credit_card_data


def valid_frame(rows=4):
    data = {column: [1] * rows for column in FEATURE_COLUMNS}
    data["AGE"] = [35] * rows
    data[TARGET_COLUMN] = [0, 1, 0, 1][:rows]
    return pd.DataFrame(data)


@pytest.mark.parametrize("target", [2, -1, 0.5, "0.5"])
def test_loader_rejects_nonbinary_or_fractional_target(target):
    frame = valid_frame()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(object)
    frame.loc[0, TARGET_COLUMN] = target

    with pytest.raises(ValueError, match="target|binary|0|1"):
        clean_credit_card_data(frame)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_loader_rejects_nonfinite_required_feature(value):
    frame = valid_frame()
    frame["LIMIT_BAL"] = frame["LIMIT_BAL"].astype(float)
    frame.loc[0, "LIMIT_BAL"] = value

    with pytest.raises(ValueError, match="finite|nonfinite|LIMIT_BAL"):
        clean_credit_card_data(frame)


def test_feature_identical_records_never_cross_splits():
    from src.train_model import _split_indices, _feature_groups
    features = pd.DataFrame({'value': np.repeat(np.arange(100), 2)})
    target = pd.Series(np.tile([0, 1], 100))
    splits = _split_indices(features, target)
    groups = _feature_groups(features)
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(200))
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(groups[splits[left]]) & set(groups[splits[right]])
    repeated = _split_indices(features, target)
    for name in splits:
        np.testing.assert_array_equal(splits[name], repeated[name])


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_prediction_accepts_unlabelled_rows_and_exports_scores(tmp_path, monkeypatch, extension):
    from sklearn.dummy import DummyClassifier
    import src.predict as predictor
    frame = valid_frame()
    model = DummyClassifier(strategy='prior').fit(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN])
    monkeypatch.setattr(predictor.joblib, 'load', lambda path: model)
    source = tmp_path / f'unlabelled.{extension}'
    unlabelled = frame.drop(columns=TARGET_COLUMN)
    if extension == "csv":
        unlabelled.to_csv(source, index=False)
    else:
        unlabelled.to_excel(source, index=False)
    destination = predictor.predict('unused', source, tmp_path / 'scores.csv')
    result = pd.read_csv(destination)
    assert len(result) == 4
    assert 'model_score' in result and 'predicted_default_probability' not in result
    assert result.risk_band.str.startswith('Score ').all()


def test_standardized_excel_headers_without_id(tmp_path):
    from src.load_data import load_credit_card_data
    source = tmp_path / 'standardized.xlsx'
    valid_frame().to_excel(source, index=False)
    assert len(load_credit_card_data(source)) == 4
