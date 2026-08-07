"""
Data pipeline for the fraud detection model.

Dataset: Kaggle "Credit Card Fraud Detection" (creditcard.csv).
Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it at data/creditcard.csv — see README.md.

Columns: Time, V1..V28 (PCA-anonymized features), Amount, Class (0=legit, 1=fraud).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from data_validation import validate_raw_data


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Download creditcard.csv from Kaggle and place it in data/."
        )
    df = pd.read_csv(RAW_DATA_PATH)
    return validate_raw_data(df)


def build_features(df: pd.DataFrame):
    """Only real feature engineering needed here: Time/Amount are on a very
    different scale than the PCA components, so they get scaled. Everything
    else is already numeric and PCA-anonymized by the dataset source.

    Returns (df, scaler) — the scaler MUST be saved alongside the model,
    since serving has to apply the exact same transform to live requests."""
    df = df.copy()
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    return df, scaler


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )