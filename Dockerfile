# syntax=docker/dockerfile:1
# Customer Churn Prediction - Streamlit app
#   docker build -t churn-app .
#   docker run --rm -p 8501:8501 churn-app
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# libgomp1: OpenMP runtime required by LightGBM / XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached as long as requirements.txt is unchanged
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Train inside the image so the pickled model always matches the installed
# library versions (the dataset is downloaded if it is not in the build context)
RUN python -m src.train --fast

# Run as an unprivileged user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
