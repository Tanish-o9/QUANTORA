# Quantora — LLM Performance & Optimization Engineering Platform
# Backend Inference & Benchmarking Container

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    QUANTORA_HOST=0.0.0.0 \
    QUANTORA_PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install PyTorch and dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy application source code
COPY . .

# Expose API server port (8000) and Streamlit fallback (8501)
EXPOSE 8000 8501

# Healthcheck to verify Quantora API readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

# Default command: launch Quantora REST API server
CMD ["python", "server.py"]
