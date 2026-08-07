import pytest
from pydantic import ValidationError

from schemas import TransactionInput

VALID_PAYLOAD = {
    "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
    "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02,
    "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19,
    "V27": 0.13, "V28": -0.02, "Amount": 149.62,
}


def test_valid_transaction_passes():
    txn = TransactionInput(**VALID_PAYLOAD)
    assert txn.Amount == 149.62


def test_negative_amount_is_rejected():
    payload = {**VALID_PAYLOAD, "Amount": -5.0}
    with pytest.raises(ValidationError):
        TransactionInput(**payload)


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "V14"}
    with pytest.raises(ValidationError):
        TransactionInput(**payload)