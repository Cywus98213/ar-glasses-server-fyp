# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with memory optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 \
    torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout=1000 -r requirements.txt && \
    pip cache purge && \
    # Clean up to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy application code
COPY . .

# Create directories for debug output
RUN mkdir -p debug_json_output recorded_audio testing_audio

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS=ignore
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TOKENIZERS_PARALLELISM=false

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["python", "ar_glasses_server.py"]
