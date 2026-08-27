"""
Consumes transaction events from Kafka and scores them with the same
ModelStore used by the REST API (serving/predict.py) — streaming and
REST ingestion share one scoring path, so there's no second copy of the
model logic to drift out of sync with the first.

Run as its own service (see docker-compose.yml: stream-consumer).
"""
import json
import os
import sys
from pathlib import Path

from kafka import KafkaConsumer

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "serving"))
from logging_config import get_logger  # noqa: E402
from predict import store  # noqa: E402

logger = get_logger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = "transactions"
GROUP_ID = "fraud-scoring-consumer"


def score_message(features: dict) -> dict:
    """Isolated from the Kafka loop so it can be unit tested without a broker."""
    is_fraud, probability = store.predict(features)
    return {
        "model_version": store.version,
        "is_fraud": is_fraud,
        "fraud_probability": round(probability, 4),
    }


def run():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    logger.info(
        "consumer starting",
        extra={"extra_fields": {"topic": TOPIC, "model_version": store.version}},
    )

    for message in consumer:
        try:
            result = score_message(message.value)
            logger.info("streamed prediction", extra={"extra_fields": result})
        except Exception as e:
            logger.error(
                "failed to score streamed transaction",
                extra={"extra_fields": {"error": str(e)}},
            )


if __name__ == "__main__":
    run()