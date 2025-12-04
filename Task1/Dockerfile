FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# scikit-learn wheels use OpenMP; install runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt --no-cache

# Copy source code
COPY . .

# Build model artifact inside the image9
RUN python train_model.py

EXPOSE 8501

# Run Streamlit and bind to all interfaces
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

