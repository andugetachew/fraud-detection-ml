"""
Drift detection via Evidently's DataDriftPreset — replaces an earlier
custom z-score check. Evidently runs a proper statistical test per
column (KS-test, PSI, Wasserstein distance, etc., chosen automatically
based on column type and sample size) instead of a single hand-rolled
mean-shift heuristic, and is the standard, recognized tool for this.
"""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Below this, per-column statistical tests (KS-test, PSI, etc.) are
# unreliable — a single point compared against a distribution will almost
# always register as "different", which is a sample-size artifact, not
# genuine drift. 30 is the conventional minimum for these tests to be
# meaningful.
MIN_SAMPLE_SIZE = 30


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    if current_df.empty:
        return {"sample_size": 0, "is_drifting": False, "drifted_features": [], "drift_share": 0.0}

    if len(current_df) < MIN_SAMPLE_SIZE:
        return {
            "sample_size": len(current_df),
            "is_drifting": False,
            "drifted_features": [],
            "drift_share": 0.0,
            "insufficient_data": True,
        }

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()

    summary = result["metrics"][0]["result"]
    per_column = result["metrics"][1]["result"]
    drifted_features = [
        column
        for column, info in per_column["drift_by_columns"].items()
        if info["drift_detected"]
    ]

    return {
        "sample_size": len(current_df),
        "is_drifting": bool(summary["dataset_drift"]),
        "drifted_features": drifted_features,
        "drift_share": round(float(summary["share_of_drifted_columns"]), 4),
    }