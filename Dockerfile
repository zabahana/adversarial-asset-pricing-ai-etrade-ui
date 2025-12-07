# GPU-enabled Dockerfile for Cloud Run with CUDA support
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python (handle existing files/symlinks)
RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    python3 --version

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy requirements and install Python dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir \
            streamlit==1.39.0 \
            pandas \
            numpy \
            plotly \
            torch \
            torchvision \
            torchaudio \
            --index-url https://download.pytorch.org/whl/cu118 \
            lightning \
            google-cloud-storage \
            google-cloud-secret-manager \
            google-cloud-logging \
            google-cloud-monitoring; \
    fi

# Install PyTorch with CUDA support if not in requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu118

# Ensure Streamlit and GCP libraries are installed
RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    google-cloud-storage \
    google-cloud-secret-manager \
    google-cloud-logging \
    google-cloud-monitoring

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p results/cached_history results/cached_features models data lightning_app/works

# Set environment variables
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port (Cloud Run will use PORT env var)
EXPOSE 8080

# Verify critical files exist
RUN test -f streamlit_app.py || (echo "ERROR: streamlit_app.py not found!" && ls -la /app && exit 1)

# Create entrypoint script with GPU detection
RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'set -e' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'PORT=${PORT:-8080}' >> /app/entrypoint.sh && \
    echo 'echo "Starting Streamlit on port $PORT"' >> /app/entrypoint.sh && \
    echo 'echo "Working directory: $(pwd)"' >> /app/entrypoint.sh && \
    echo 'echo "GPU Detection:"' >> /app/entrypoint.sh && \
    echo 'if command -v nvidia-smi &> /dev/null; then' >> /app/entrypoint.sh && \
    echo '    echo "NVIDIA SMI found - checking GPU availability..."' >> /app/entrypoint.sh && \
    echo '    nvidia-smi || echo "GPU not available in this container"' >> /app/entrypoint.sh && \
    echo 'else' >> /app/entrypoint.sh && \
    echo '    echo "nvidia-smi not found - running on CPU"' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo 'if [ ! -f /app/streamlit_app.py ]; then' >> /app/entrypoint.sh && \
    echo '    echo "ERROR: streamlit_app.py not found!"' >> /app/entrypoint.sh && \
    echo '    ls -la' >> /app/entrypoint.sh && \
    echo '    exit 1' >> /app/entrypoint.sh && \
    echo 'fi' >> /app/entrypoint.sh && \
    echo 'exec streamlit run streamlit_app.py \' >> /app/entrypoint.sh && \
    echo '    --server.port=$PORT \' >> /app/entrypoint.sh && \
    echo '    --server.address=0.0.0.0 \' >> /app/entrypoint.sh && \
    echo '    --server.headless=true \' >> /app/entrypoint.sh && \
    echo '    --server.enableCORS=false \' >> /app/entrypoint.sh && \
    echo '    --server.enableXsrfProtection=false \' >> /app/entrypoint.sh && \
    echo '    --server.fileWatcherType=none' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Run Streamlit app using entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

