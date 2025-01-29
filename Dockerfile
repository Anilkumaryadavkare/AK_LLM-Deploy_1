# Use lighter base image
FROM python:3.9-slim-bullseye

# Set cache environment variables
ENV PIP_CACHE_DIR=/tmp/pip-cache \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME=/tmp/huggingface-cache

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files last
COPY . .

# Start command with health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "web_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]