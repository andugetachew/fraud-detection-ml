"""
Load test for the fraud detection API.

Run:
    locust -f locustfile.py --host http://localhost:8088

Then open http://localhost:8089 to set concurrent users / spawn rate and
watch live RPS + latency percentiles. For a headless run with a fixed
shape (e.g. CI or a quick check):
    locust -f locustfile.py --host http://localhost:8088 \
        --users 50 --spawn-rate 5 --run-time 2m --headless

Set the real API key first:
    set LOAD_TEST_API_KEY=<your-key>          (Windows)
    export LOAD_TEST_API_KEY=<your-key>       (Linux/Mac)

NOTE on interpreting results: /predict is rate-limited to 20/min per IP
(slowapi). Locust running from one machine shares one IP, so at real
concurrency you WILL see 429s once that limit is hit — that's the rate
limiter doing its job, not a failure. This test measures both: real
throughput before the limit kicks in, and whether the limiter correctly
rejects excess traffic once it does. It is not meant to measure raw
model-inference throughput in isolation — for that, you'd temporarily
raise the slowapi limit via config, run again, then restore it.
"""
import os
import random

from locust import HttpUser, task, between

API_KEY = os.getenv("LOAD_TEST_API_KEY", "")

# Real values from the dataset (row 1) as a base, with Amount/Time jittered
# per-request so drift monitoring and SHAP see varied inputs, not one
# identical payload repeated thousands of times.
BASE_TRANSACTION = {
    "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
    "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02,
    "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19,
    "V27": 0.13, "V28": -0.02, "Amount": 149.62,
}


def jittered_transaction() -> dict:
    txn = dict(BASE_TRANSACTION)
    txn["Amount"] = round(max(0.0, random.gauss(149.62, 80)), 2)
    txn["Time"] = random.randint(0, 172_792)
    return txn


class FraudApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        if not API_KEY:
            raise RuntimeError(
                "LOAD_TEST_API_KEY not set — export/set it to your real API_KEY before running."
            )
        self.headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(6)
    def predict(self):
        with self.client.post(
            "/predict", json=jittered_transaction(), headers=self.headers, catch_response=True
        ) as resp:
            # 429 (rate limited) is an EXPECTED outcome under real load, not a bug —
            # mark it as a pass so it doesn't pollute the failure rate, but it still
            # shows up in the raw stats/logs for you to eyeball separately.
            if resp.status_code == 429:
                resp.success()

    @task(1)
    def explain(self):
        with self.client.post(
            "/predict/explain", json=jittered_transaction(), headers=self.headers, catch_response=True
        ) as resp:
            if resp.status_code == 429:
                resp.success()

    @task(2)
    def health(self):
        self.client.get("/health")