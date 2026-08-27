"""
Simulates a live transaction stream by replaying creditcard.csv rows onto a
Kafka topic, one at a time. This is the streaming counterpart to the existing
REST /predict endpoint: instead of a client calling the API per-transaction,
transactions arrive as events and get scored by streaming/consumer.py.

Run as its own service (see docker-compose.yml: stream-producer).
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from config import RAW_DATA_PATH, TARGET_COLUMN  # noqa: E402
from logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = "transactions"
# Simulated arrival rate. Real transactions wouldn't be evenly spaced, but a
# fixed delay is enough to demonstrate streaming ingestion vs. batch replay.
STREAM_DELAY_SECONDS = float(os.getenv("STREAM_DELAY_SECONDS", "0.5"))


def feature_columns(df: pd.DataFrame) -> list[str]:
    # Class (the fraud label) is dropped from the payload. A real ingestion
    # pipeline scores transactions before the outcome is known — sending the
    # label would be leaking the answer into the "unscored" event.
    return [c for c in df.columns if c != TARGET_COLUMN]


def run():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    df = pd.read_csv(RAW_DATA_PATH)
    feature_cols = feature_columns(df)

    logger.info(
        "producer starting",
        extra={"extra_fields": {"topic": TOPIC, "rows": len(df), "bootstrap": KAFKA_BOOTSTRAP_SERVERS}},
    )

    for i, row in df.iterrows():
        payload = row[feature_cols].to_dict()
        producer.send(TOPIC, payload)

        if i % 100 == 0:
            producer.flush()
            logger.info("produced batch", extra={"extra_fields": {"row_index": int(i)}})

        time.sleep(STREAM_DELAY_SECONDS)

    producer.flush()
    logger.info("producer finished", extra={"extra_fields": {"rows_sent": len(df)}})


if __name__ == "__main__":
    run()