import numpy as np
import pandas as pd
from drift import compute_drift


def make_reference_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Amount": rng.normal(88, 250, n),
        "Time": rng.normal(50000, 28000, n),
    })


def test_no_drift_when_current_matches_reference():
    reference_df = make_reference_df()
    current_df = make_reference_df(n=200, seed=1)  # same distribution, different sample
    result = compute_drift(reference_df, current_df)
    assert result["sample_size"] == 200
    assert result["is_drifting"] is False


def test_drift_flagged_when_distribution_shifts():
    reference_df = make_reference_df()
    rng = np.random.default_rng(2)
    current_df = pd.DataFrame({
        "Amount": rng.normal(5000, 250, 200),  # Amount shifted far from reference
        "Time": rng.normal(50000, 28000, 200),  # Time unchanged
    })
    result = compute_drift(reference_df, current_df)
    assert result["is_drifting"] is True
    assert "Amount" in result["drifted_features"]


def test_empty_current_data_returns_no_drift():
    reference_df = make_reference_df()
    result = compute_drift(reference_df, pd.DataFrame())
    assert result == {"sample_size": 0, "is_drifting": False, "drifted_features": [], "drift_share": 0.0}


def test_drift_share_is_between_zero_and_one():
    reference_df = make_reference_df()
    current_df = make_reference_df(n=200, seed=1)
    result = compute_drift(reference_df, current_df)
    assert 0.0 <= result["drift_share"] <= 1.0


def test_tiny_sample_does_not_falsely_flag_drift():
    """Regression test: a single live prediction compared against a
    distribution will trivially fail statistical tests on nearly every
    column — that's a sample-size artifact, not real drift."""
    reference_df = make_reference_df()
    current_df = make_reference_df(n=1, seed=1)
    result = compute_drift(reference_df, current_df)
    assert result["is_drifting"] is False
    assert result["drifted_features"] == []
    assert result["insufficient_data"] is True