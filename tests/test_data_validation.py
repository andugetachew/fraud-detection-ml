import numpy as np
import pandas as pd
import pytest
from data_validation import validate_raw_data


def make_valid_df(n=50):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = rng.integers(0, 100_000, n).astype(float)
    df["Amount"] = rng.exponential(50, n)
    df["Class"] = rng.choice([0, 1], n)
    return df


def test_valid_dataset_passes():
    df = make_valid_df()
    validated = validate_raw_data(df)
    assert len(validated) == len(df)


def test_negative_amount_is_rejected():
    df = make_valid_df()
    df.loc[0, "Amount"] = -10.0
    with pytest.raises(ValueError):
        validate_raw_data(df)


def test_negative_time_is_rejected():
    df = make_valid_df()
    df.loc[0, "Time"] = -1.0
    with pytest.raises(ValueError):
        validate_raw_data(df)


def test_invalid_class_label_is_rejected():
    df = make_valid_df()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError):
        validate_raw_data(df)


def test_missing_column_is_rejected():
    df = make_valid_df().drop(columns=["V14"])
    with pytest.raises(ValueError):
        validate_raw_data(df)


def test_unexpected_extra_column_is_rejected():
    df = make_valid_df()
    df["unexpected_column"] = 1
    with pytest.raises(ValueError):
        validate_raw_data(df)


def test_null_value_is_rejected():
    df = make_valid_df()
    df.loc[0, "V1"] = None
    with pytest.raises(ValueError):
        validate_raw_data(df)