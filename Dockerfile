FROM python:3.12-slim AS builder

WORKDIR /app

# gcc only needed to build some wheels — stays in this stage, never ships
# in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
# Build-time proof the install actually landed in this venv, not
# wherever PATH happened to resolve pip to.
RUN /opt/venv/bin/pip show uvicorn


FROM python:3.11-slim

# libgomp1 is a runtime dependency of xgboost — needed here too, gcc is not.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY training/ ./training/
COPY serving/ ./serving/
RUN chown -R appuser:appuser /app /opt/venv

USER appuser
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app/serving

# Build-time sanity check: fail loudly here (clear build log) rather than
# with a vague runtime "not found" if the venv copy/permissions are ever wrong.
RUN /opt/venv/bin/uvicorn --version

CMD ["sh", "-c", "/opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"]