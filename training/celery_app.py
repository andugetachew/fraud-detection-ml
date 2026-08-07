from celery import Celery
from config import CELERY_TASK_ALWAYS_EAGER, REDIS_URL

app = Celery("fraud_detection_ml", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_always_eager=CELERY_TASK_ALWAYS_EAGER,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

app.conf.imports = ("tasks",)