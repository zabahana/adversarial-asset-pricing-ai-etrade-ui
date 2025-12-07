#!/bin/bash
# Test Docker container locally before deploying

echo "Building Docker image..."
docker build -t test-streamlit-local .

echo ""
echo "Starting container on port 8501..."
echo "Open http://localhost:8501 in your browser once it starts"
echo ""
echo "Press Ctrl+C to stop"
echo ""

docker run --rm -p 8501:8501 -e PORT=8501 test-streamlit-local

