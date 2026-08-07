"""
Run with: python train.py
Produces a versioned model artifact under artifacts/ and registers it
in the model_versions table. Registering is NOT the same as activating —
activation (which version serving actually uses) is a separate, explicit
step via `python train.py --activate` or registry.activate_version().
"""
import argparse
from datetime import datetime, timezone

import joblib
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from config import ARTIFACTS_DIR, RANDOM_STATE
from data_pipeline import load_raw_data, build_features, split_data
from registry import register_version, activate_version
from logging_config import get_logger

logger = get_logger(__name__)

REFERENCE_SAMPLE_SIZE = 5000  # enough for Evidently's tests to be statistically meaningful, small enough to stay cheap


def train_model(X_train, y_train) -> XGBClassifier:
    # Fraud datasets are heavily imbalanced (~0.17% positive class here).
    # scale_pos_weight is the cheap, effective fix — no need for SMOTE
    # unless the imbalance is even more extreme than this.
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def main(activate: bool):
    logger.info("Loading data")
    raw_df = load_raw_data()
    df, scaler = build_features(raw_df)
    X_train, X_test, y_train, y_test = split_data(df)

    logger.info("Training model", extra={"extra_fields": {"n_train_rows": len(X_train)}})
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    logger.info("Evaluated on holdout", extra={"extra_fields": {"metrics": metrics}})

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    artifact_path = ARTIFACTS_DIR / f"model_{version}.joblib"
    # Model and scaler saved together — serving needs both to reproduce
    # training-time preprocessing exactly.
    joblib.dump({"model": model, "scaler": scaler}, artifact_path)

    # Reference sample saved on RAW (pre-scaling) values, matching what
    # /predict logs from live requests — Evidently needs the actual
    # distribution here, not scaled training values or summary stats.
    raw_X_train = raw_df.loc[X_train.index].drop(columns=["Class"])
    reference_sample = raw_X_train.sample(
        n=min(REFERENCE_SAMPLE_SIZE, len(raw_X_train)), random_state=RANDOM_STATE
    )
    reference_data_path = ARTIFACTS_DIR / f"reference_{version}.csv"
    reference_sample.to_csv(reference_data_path, index=False)

    register_version(
        version=version,
        artifact_path=str(artifact_path),
        metrics=metrics,
        reference_data_path=str(reference_data_path),
    )
    logger.info(
        "Registered model version",
        extra={"extra_fields": {"version": version, "artifact_path": str(artifact_path)}},
    )

    if activate:
        activate_version(version)
        logger.info("Activated version", extra={"extra_fields": {"version": version}})
    else:
        logger.info(
            "Version not activated — review metrics before activating",
            extra={"extra_fields": {"version": version}},
        )

    return {"version": version, "metrics": metrics, "activated": activate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activate", action="store_true",
        help="Activate this version immediately after training (skip manual review)"
    )
    args = parser.parse_args()
    main(activate=args.activate)