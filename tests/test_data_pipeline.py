import numpy as np
import pandas as pd
import pytest
from data_pipeline import build_features, load_raw_data, split_data


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 200
    df = pd.DataFrame(np.random.randn(n, 28), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.random.randint(0, 100000, n)
    df["Amount"] = np.random.exponential(50, n)
    df["Class"] = np.random.choice([0, 1], n, p=[0.98, 0.02])
    return df


def test_build_features_scales_time_and_amount(sample_df):
    scaled_df, scaler = build_features(sample_df)
    # Scaled columns should be ~standard-normal
    assert abs(scaled_df["Time"].mean()) < 0.01
    assert abs(scaled_df["Amount"].mean()) < 0.01
    # Scaler must be returned — it has to be saved alongside the model,
    # or serving can't reproduce this transform at inference time.
    assert hasattr(scaler, "transform")


def test_build_features_does_not_mutate_input(sample_df):
    original_amount = sample_df["Amount"].copy()
    build_features(sample_df)
    pd.testing.assert_series_equal(sample_df["Amount"], original_amount)


def test_split_data_is_stratified(sample_df):
    scaled_df, _ = build_features(sample_df)
    X_train, X_test, y_train, y_test = split_data(scaled_df)

    assert len(X_train) + len(X_test) == len(scaled_df)
    assert "Class" not in X_train.columns
    # Stratified split should keep the fraud rate roughly consistent
    # between train and test, not skew it toward one side.
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_load_raw_data_missing_file_raises(monkeypatch, tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr("data_pipeline.RAW_DATA_PATH", missing_path)
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data()


def test_load_raw_data_reads_and_validates_existing_file(monkeypatch, tmp_path, sample_df):
    csv_path = tmp_path / "creditcard.csv"
    sample_df.to_csv(csv_path, index=False)
    monkeypatch.setattr("data_pipeline.RAW_DATA_PATH", csv_path)

    result = load_raw_data()

    assert len(result) == len(sample_df)
    assert set(result.columns) == set(sample_df.columns)