FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# psutil needs gcc to compile on Linux
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache: pip install only re-runs when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY sla_defaults.json .
COPY pyproject.toml .
COPY model/artifacts/ ./model/artifacts/

# BYOD custom models are runtime-supplied by design and must not be baked in
RUN mkdir -p /root/.cetp

EXPOSE 8000

# curl is not in slim; use urllib instead
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
