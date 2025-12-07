#!/bin/bash
set -e

# Use PORT environment variable if provided, otherwise default to 8501
PORT=${PORT:-8501}
export PORT

# Ensure Python is available
which python3 || which python || exit 1

# Change to app directory (if running in Docker) or use current directory
if [ -d "/app" ]; then
    cd /app
fi

# Start Streamlit app
exec streamlit run streamlit_app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --logger.level=error

