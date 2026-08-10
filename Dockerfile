FROM python:3.12-slim AS builder

WORKDIR /app

# gcc only needed to build some wheels — stays in this stage, never ships
# in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --target installs pure package files only — no interpreter binary, no
# venv, no symlinks. This avoids the whole class of "copied Python breaks
# in the next stage" issues (broken symlinks, mismatched shared libraries)
# since we never copy a Python interpreter between stages at all.
RUN pip install --no-cache-dir --target=/deps -r requirements.txt
# Debug: confirm uvicorn actually landed in /deps before we copy it anywhere.
RUN ls -la /deps | grep -i uvicorn || (echo "uvicorn MISSING from /deps" && ls -la /deps && exit 1)


FROM python:3.12-slim

# libgomp1 is a runtime dependency of xgboost — needed here too, gcc is not.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

# Copy pure package files straight into this stage's OWN Python's
# site-packages. This stage's python/libpython are guaranteed consistent
# with each other since they come from one image, not two.
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages

WORKDIR /app
COPY training/ ./training/
COPY serving/ ./serving/
RUN chown -R appuser:appuser /app

USER appuser
WORKDIR /app/serving

# Build-time sanity check: fail loudly here (clear build log) rather than
# with a vague runtime failure if the copy ever goes wrong. Module
# invocation (-m) is used everywhere instead of the uvicorn/celery console
# scripts, since --target installs don't generate those entry-point
# scripts at all.
RUN python -m uvicorn --version

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"]