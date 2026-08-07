# Architecture

End-to-end fraud detection system: training, serving, and monitoring in
one repo, containerized, backed by managed Postgres and Redis.

## System diagram

```
                                   ┌─────────────────────┐
                                   │   Neon PostgreSQL    │
                                   │  (model_versions,     │
                                   │   prediction_logs)     │
                                   └──────────┬────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
              ┌─────▼─────┐            ┌──────▼──────┐           ┌──────▼──────┐
              │   worker   │            │     api      │           │    beat     │
              │  (Celery)  │            │  (FastAPI,    │           │  (Celery    │
              │            │            │  4 processes) │           │  scheduler) │
              └─────┬──────┘            └──────┬───────┘           └──────┬──────┘
                    │                          │                          │
                    │      ┌───────────────────┴──────────┐               │
                    │      │                               │               │
                    │  client requests                prediction          │
                    │  (/predict, /predict/explain,   log write            │
                    │   /model/*, /health, /metrics)  (async task)         │
                    │                                                      │
                    └──────────────────┬───────────────────────────────────┘
                                       │
                                 ┌─────▼──────┐
                                 │    Redis    │
                                 │  (broker +   │
                                 │   backend)   │
                                 └────────────┘
```

All three services (`api`, `worker`, `beat`) share the same Docker image
and codebase, differing only in their entrypoint command. `redis` is
local to the Docker network; `Neon` is external, managed Postgres.

## Components

### `training/`
Data ingestion → validation → feature engineering → training →
evaluation → registration. Run manually (`python train.py`) or as a
Celery background job (`tasks.train_task`).

- `data_pipeline.py` — loads the raw CSV, scales `Time`/`Amount`
  (`StandardScaler`), splits train/test
- `data_validation.py` — Pandera schema validation on the raw dataset
  before it reaches training (types, ranges, nulls, unexpected columns)
- `train.py` — trains an `XGBClassifier` (`scale_pos_weight` handles the
  ~0.17% class imbalance), evaluates on a holdout set, saves the model
  **and** its fitted scaler together in one artifact (they must travel
  together — serving needs the exact same transform used at training
  time), and saves a reference sample of raw training rows for drift
  comparison later
- `registry.py` — SQLAlchemy models (`ModelVersion`, `PredictionLog`)
  and the functions that read/write them. One `model_versions` row per
  trained version; exactly one can be `is_active` at a time
- `drift.py` — Evidently `DataDriftPreset`-based drift check, comparing
  recent live prediction inputs against the reference sample
- `tasks.py` / `celery_app.py` — background jobs: async training,
  async prediction logging (moves the Neon write off the request path),
  scheduled hourly drift checks

### `serving/`
FastAPI app exposing the model over HTTP. Runs as `api`, 4 uvicorn
worker processes for real concurrency.

- `main.py` — routes, auth/rate-limit wiring, Prometheus instrumentation,
  request logging middleware
- `predict.py` — `ModelStore`: holds the active model+scaler in memory,
  runs inference, fires prediction logging as a background task
- `schemas.py` — Pydantic request/response models
- `auth.py` — API key check (fails closed if unconfigured)

### `data/`, `artifacts/`
Mounted volumes, not baked into the Docker image. `data/creditcard.csv`
is the training set (not committed — see README for the download link).
`artifacts/` holds trained model files and reference-sample CSVs; it's
where the local SQLite fallback registry would also live if `DATABASE_URL`
isn't set.

### `tests/`
40 tests, 95%+ coverage on everything except thin DB/config/CLI wiring
(deliberately excluded from the coverage denominator — see
`pyproject.toml`). API-layer tests (`test_api.py`) exercise the real
FastAPI app via `TestClient`, with the DB, model artifact, Celery
broker, and SHAP explainer mocked at the module boundary — no live
Neon/Redis needed to run the suite.

### `.github/workflows/ci.yml`
lint → test (with coverage gate) → docker build → deploy (on merge to
`main` only).

### `load_testing/`
Locust script for `/predict`/`/predict/explain`/`/health`, with notes
on interpreting results against the configured rate limits.

## Key design decisions

**Model + scaler saved together.** Scaling `Time`/`Amount` is fit at
training time; if serving used a different (or freshly-refit) scaler,
predictions would be silently wrong. They're bundled in one artifact so
that can't happen.

**Reference sample, not summary stats, for drift.** Early version stored
just mean/std per feature. Evidently's statistical tests (KS-test, PSI)
need actual reference *rows* to compare distributions properly — summary
stats aren't enough for a real test.

**Prediction logging is asynchronous.** Originally `/predict` wrote to
Neon synchronously before responding. Load testing revealed this made
p95 latency ~11 seconds under concurrent load (`docker stats` showed
near-zero CPU during that time — an I/O-wait signature, not compute).
Fixed by moving the write to a Celery background task; verified via
load test afterward: p95 dropped from 11,000ms → 56ms.

**`DEBUG` controls Celery eager mode.** `DEBUG=true` → tasks run
synchronously in-process (no worker needed, convenient for local
poking). `DEBUG=false` → tasks genuinely queue to Redis and run on the
worker. (This relationship was inverted by a real bug for a while —
`CELERY_TASK_ALWAYS_EAGER = not DEBUG` — silently defeating the async
fix above; caught via load testing and locked in with a regression test,
`tests/test_config.py`.)

**Auth fails closed.** If `API_KEY` isn't set on the server at all,
protected endpoints reject every request (`503`) rather than silently
running unauthenticated.

**Multi-worker model reload is best-effort, not guaranteed.** `api` runs
4 separate processes, each with its own in-memory model. `/model/reload`
only reloads whichever process handles that request. Documented rather
than silently wrong — `docker compose restart api` is the reliable path
after activating a new model version.

**Drift checks skip below a minimum sample size.** Statistical tests on
1-2 data points are meaningless (a single point vs. a distribution will
almost always register as "different"). Below 30 recent predictions,
`/model/drift` returns `insufficient_data: true` instead of a false
alarm.

## What's deliberately not covered by unit tests

`registry.py`'s DB functions, `celery_app.py`, `evaluate.py` (a CLI
tool), and `logging_config.py` are excluded from the coverage
denominator — testing them meaningfully would mean mocking SQLAlchemy
into uselessness, or testing config wiring rather than logic. The
`api_client` fixture in `tests/conftest.py` tests the real request-
handling logic in `main.py`/`predict.py` instead, with those thin
boundaries mocked one layer down.