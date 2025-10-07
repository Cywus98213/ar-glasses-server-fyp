# Use Python 3.10 slim image for memory efficiency
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with memory optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    # Install PyTorch CPU-only (much smaller)
    pip install --no-cache-dir --timeout=1000 \
    torch==2.0.1+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    # Install other dependencies
    pip install --no-cache-dir --timeout=1000 -r requirements.txt && \
    # Aggressive cleanup to save memory
    pip cache purge && \
    rm -rf /root/.cache/pip && \
    find /usr/local/lib/python3.10 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10 -name "__pycache__" -type d -exec rm -rf {} + || true

# Copy application code
COPY . .

# Create directories for debug output and models
RUN mkdir -p debug_json_output recorded_audio testing_audio pretrained_models

# Expose port
EXPOSE 8000

# Set memory-efficient environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS=ignore
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMBA_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# No health check - Render will detect WebSocket service automatically

# Run the full server for Render
CMD ["python", "ar_glasses_server.py"]
