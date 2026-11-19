# One image for the API + website, the MLflow server and the training jobs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# curl is used by the docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml .
COPY src/ src/
COPY api/ api/
COPY web/ web/

# run as a normal user, not root
RUN useradd --create-home --uid 1000 appuser && mkdir -p data models logs metrics reports mlflow \
    && chown -R appuser /app
USER appuser

EXPOSE 8000 5000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
