#!/bin/bash
# Set OpenAI API Key for Cloud Run service

set -e

SERVICE_NAME="adversarial-asset-pricing-ai"
REGION="us-central1"

if [ -z "$1" ]; then
    echo "Usage: ./set_openai_key.sh YOUR_OPENAI_API_KEY"
    echo ""
    echo "Example:"
    echo "  ./set_openai_key.sh sk-proj-..."
    exit 1
fi

OPENAI_API_KEY="$1"

echo "Setting OpenAI API key for Cloud Run service..."

gcloud run services update ${SERVICE_NAME} \
  --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY}" \
  --region ${REGION}

echo "✅ OpenAI API key set successfully!"
echo ""
echo "The service will now use LLM-powered summaries."

