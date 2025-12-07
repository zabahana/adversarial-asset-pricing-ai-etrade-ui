#!/bin/bash
# Test Docker image locally before deploying

echo "Testing Docker image locally..."

# Build the image
docker build -t test-streamlit .

echo ""
echo "Starting container on port 8502..."
docker run -p 8502:8501 -e PORT=8501 test-streamlit &

sleep 10

echo ""
echo "Testing health check..."
curl -f http://localhost:8502/_stcore/health || echo "Health check failed"

echo ""
echo "Container should be running. Open http://localhost:8502 in your browser."
echo "Press Ctrl+C to stop when done testing."
echo ""
echo "Container logs:"
docker ps --filter ancestor=test-streamlit --format "{{.ID}}"

wait

