import numpy as np
import pandas as pd

import train
from train import train_model, evaluate_model


def make_synthetic_split(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.choice([0, 1], n, p=[0.9, 0.1]))
    split = n * 3 // 4
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def test_train_model_handles_imbalanced_classes():
    X_train, _, y_train, _ = make_synthetic_split()
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict_proba")


def test_evaluate_model_returns_expected_metric_keys():
    X_train, X_test, y_train, y_test = make_synthetic_split()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert set(metrics.keys()) == {"precision", "recall", "f1", "roc_auc"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_main_orchestrates_pipeline_and_registers_version(monkeypatch, tmp_path):
    """Exercises main() end-to-end with the DB/file boundaries mocked —
    this is what actually happens, just without touching real Neon or
    the real 227k-row dataset."""
    raw_df = pd.DataFrame({
        **{f"V{i}": np.random.default_rng(i).standard_normal(300) for i in range(1, 29)},
        "Time": np.random.default_rng(99).integers(0, 100_000, 300).astype(float),
        "Amount": np.random.default_rng(98).exponential(50, 300),
        "Class": np.random.default_rng(97).choice([0, 1], 300, p=[0.9, 0.1]),
    })

    monkeypatch.setattr(train, "load_raw_data", lambda: raw_df)
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)

    registered = {}
    monkeypatch.setattr(
        train, "register_version",
        lambda **kwargs: registered.update(kwargs)
    )
    activated = {}
    monkeypatch.setattr(
        train, "activate_version",
        lambda version: activated.setdefault("version", version)
    )

    result = train.main(activate=True)

    assert "version" in result
    assert set(result["metrics"].keys()) == {"precision", "recall", "f1", "roc_auc"}
    assert result["activated"] is True
    assert registered["version"] == result["version"]
    assert (tmp_path / f"model_{result['version']}.joblib").exists()
    assert (tmp_path / f"reference_{result['version']}.csv").exists()
    assert activated["version"] == result["version"]