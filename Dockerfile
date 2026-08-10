FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ ./training/
COPY serving/ ./serving/
RUN chown -R appuser:appuser /app

USER appuser
WORKDIR /app/serving

# Build-time sanity check: fail loudly here, in the build log, rather than
# with a vague runtime failure later.
RUN python -m uvicorn --version

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"]