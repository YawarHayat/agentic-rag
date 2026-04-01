FROM python:3.11-slim

# System dependencies for ChromaDB / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY *.py .
COPY .env.example .env.example

# Streamlit port
EXPOSE 8501

# Streamlit config — disable telemetry, set server options
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "streamlit_app.py"]
