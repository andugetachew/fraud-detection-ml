FROM python:3.12-slim AS builder

WORKDIR /app

# gcc only needed to build some wheels — stays in this stage, never ships
# in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.11-slim

# libgomp1 is a runtime dependency of xgboost — needed here too, gcc is not.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY training/ ./training/
COPY serving/ ./serving/
RUN chown -R appuser:appuser /app

USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH

CMD ["sh", "-c", "cd /app/serving && exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"]