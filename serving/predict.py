import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap

# Reuse the same registry module training/ already uses, instead of
# duplicating the "which version is active" logic here.
sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from registry import get_active_version  # noqa: E402
from tasks import log_prediction_task  # noqa: E402

FEATURE_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)


class ModelStore:
    """Holds the currently loaded model+scaler in memory. Call reload()
    after activating a new version so serving picks it up without a
    container restart."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.version = None
        self.metrics = None
        self.reference_data_path = None
        self.created_at = None
        self._explainer = None  # built lazily — only needed if /predict/explain is actually called
        self.reload()

    def reload(self):
        active = get_active_version()
        if active is None:
            raise RuntimeError(
                "No active model version found. Run training/train.py "
                "then training/evaluate.py --activate <version> first."
            )
        artifact = joblib.load(active.artifact_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.version = active.version
        self.metrics = json.loads(active.metrics)
        self.reference_data_path = active.reference_data_path
        self.created_at = str(active.created_at)
        self._explainer = None  # invalidate — next explain() rebuilds it for the new model

    def _scaled_row(self, features: dict) -> pd.DataFrame:
        row = pd.DataFrame([features])[FEATURE_COLUMNS]
        row[["Time", "Amount"]] = self.scaler.transform(row[["Time", "Amount"]])
        return row

    def predict(self, features: dict) -> tuple[bool, float]:
        row = self._scaled_row(features)
        proba = float(self.model.predict_proba(row)[0, 1])
        is_fraud = proba >= 0.5
        # .delay() only publishes to Redis — doesn't wait for the worker
        # to actually process it, so this returns near-instantly even
        # though the Neon write itself might take 100s of ms.
        log_prediction_task.delay(self.version, features, proba, is_fraud)
        return is_fraud, proba

    def explain(self, features: dict, top_n: int = 10) -> list[dict]:
        """Returns the top_n features driving this specific prediction,
        via SHAP TreeExplainer (fast — no background dataset needed for
        tree models, unlike KernelExplainer)."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)

        row = self._scaled_row(features)
        explanation = self._explainer(row)
        # XGBoost binary classifier -> a single output per row.
        shap_values = explanation.values[0]

        contributions = [
            {"feature": col, "shap_value": round(float(val), 5), "value": float(row[col].iloc[0])}
            for col, val in zip(FEATURE_COLUMNS, shap_values)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return contributions[:top_n]


store = ModelStore()