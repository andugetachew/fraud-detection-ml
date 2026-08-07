"""
Validates the raw dataset BEFORE it reaches feature engineering or
training. This is deliberately separate from schemas.py (which validates
individual API requests at inference time) — this validates the whole
training dataset's shape and quality at ingestion time.
"""
import pandera as pa
from pandera import Check, Column, DataFrameSchema

_V_COLUMNS = {f"V{i}": Column(float, nullable=False) for i in range(1, 29)}

RAW_SCHEMA = DataFrameSchema(
    {
        "Time": Column(float, Check.ge(0), nullable=False),
        **_V_COLUMNS,
        "Amount": Column(float, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=True,   # no unexpected extra columns — a schema change upstream should fail loudly
    coerce=True,   # int/float columns read from CSV as object types get coerced, not silently skipped
)


def validate_raw_data(df):
    """Raises pandera.errors.SchemaError with a specific, actionable
    message if the dataset doesn't match what training expects. Called
    right after loading, before any feature engineering happens."""
    try:
        return RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(
            f"Raw dataset failed validation ({len(exc.failure_cases)} issue(s)):\n"
            f"{exc.failure_cases.to_string()}"
        ) from exc