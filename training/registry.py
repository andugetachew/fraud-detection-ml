"""
Model registry — deliberately simple: one table, no external MLflow server.
Serving reads from this same table to know which artifact file to load.
"""
import json
from datetime import datetime, timezone

from config import DATABASE_URL
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
# pool_pre_ping: tests each connection with a lightweight query before reuse,
# transparently reconnecting if it's dead. Required for Neon — its serverless
# tier auto-suspends the compute after inactivity, closing idle connections
# server-side, and SQLAlchemy's default pool doesn't validate before reuse.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    version = Column(String, unique=True, nullable=False)
    artifact_path = Column(String, nullable=False)
    metrics = Column(Text, nullable=False)               # JSON string: precision/recall/f1/roc_auc
    reference_data_path = Column(String, nullable=False)  # CSV of raw training features — Evidently's reference dataset
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionLog(Base):
    """One row per /predict call. Monitoring compares recent rows here
    against the active version's reference_data_path to flag drift."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    model_version = Column(String, nullable=False)
    features = Column(Text, nullable=False)   # JSON string: the request's feature values
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(engine)


def register_version(version: str, artifact_path: str, metrics: dict, reference_data_path: str) -> ModelVersion:
    init_db()
    session = SessionLocal()
    try:
        entry = ModelVersion(
            version=version,
            artifact_path=artifact_path,
            metrics=json.dumps(metrics),
            reference_data_path=reference_data_path,
            is_active=False,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry
    finally:
        session.close()


def activate_version(version: str):
    """Marks one version active and deactivates all others. Serving always
    reads the single active row — this is the whole "deployment" step."""
    init_db()
    session = SessionLocal()
    try:
        session.query(ModelVersion).update({ModelVersion.is_active: False})
        target = session.query(ModelVersion).filter_by(version=version).first()
        if not target:
            raise ValueError(f"No model version '{version}' found in registry")
        target.is_active = True
        session.commit()
    finally:
        session.close()


def get_active_version() -> ModelVersion | None:
    init_db()
    session = SessionLocal()
    try:
        return session.query(ModelVersion).filter_by(is_active=True).first()
    finally:
        session.close()


def list_versions() -> list[ModelVersion]:
    init_db()
    session = SessionLocal()
    try:
        return session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    finally:
        session.close()


def log_prediction(model_version: str, features: dict, fraud_probability: float, is_fraud: bool):
    init_db()
    session = SessionLocal()
    try:
        session.add(PredictionLog(
            model_version=model_version,
            features=json.dumps(features),
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
        ))
        session.commit()
    finally:
        session.close()


def get_recent_predictions(model_version: str, limit: int = 500) -> list[PredictionLog]:
    init_db()
    session = SessionLocal()
    try:
        return (
            session.query(PredictionLog)
            .filter_by(model_version=model_version)
            .order_by(PredictionLog.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()