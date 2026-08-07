import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Same env var name/pattern as the Django projects (dj_database_url-compatible URL).
# Local dev default lives inside artifacts/, which is a mounted volume —
# NOT the container's own filesystem, which gets wiped on `docker compose down`.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ARTIFACTS_DIR / 'local_registry.db'}")

RAW_DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Class"

# Same isolated-db-number pattern as the other projects sharing one Upstash
# instance in prod (redis://.../0, /1, /2...). Local dev points at the
# docker-compose redis service instead.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/4")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
CELERY_TASK_ALWAYS_EAGER = DEBUG

# Required to call protected serving endpoints (/predict, /model/reload).
# No insecure default — if this isn't set, those endpoints refuse all
# requests rather than silently running unauthenticated.
API_KEY = os.getenv("API_KEY")
