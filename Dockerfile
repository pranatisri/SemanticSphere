FROM python:3.11-slim

WORKDIR /app

# System deps (kept minimal); git is sometimes required by HF model downloads in certain envs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Precompute is optional; you can comment this out if you prefer lazy build on first query.
# RUN python scripts/precompute.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

