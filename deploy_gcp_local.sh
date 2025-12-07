#!/bin/bash
# GCP Cloud Run Deployment Script - Build Locally to Avoid Cloud Build Permissions

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="adversarial-asset-pricing-ai"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Deploying Streamlit app to GCP Cloud Run (Local Build)..."
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"

# Step 1: Set GCP project
echo "Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Step 2: Enable required APIs
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com || true
gcloud services enable containerregistry.googleapis.com || true

# Step 3: Configure Docker for GCR
echo "Configuring Docker for Google Container Registry..."
gcloud auth configure-docker --quiet || true

# Step 4: Build Docker image locally for linux/amd64 (required for Cloud Run)
echo "Building Docker image locally for linux/amd64 platform..."
docker buildx build --platform linux/amd64 -t ${IMAGE_NAME} --load .

# Step 5: Push image to GCR
echo "Pushing Docker image to Google Container Registry..."
docker push ${IMAGE_NAME}

# Step 6: Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 10 \
  --port 8501 \
  --min-instances 0 \
  --concurrency 10 \
  --set-env-vars "ALPHA_VANTAGE_API_KEY=5X8N02ORS7PVFFZ4" \
  --execution-environment=gen2 \
  --cpu-boost

# Step 7: Get the service URL
echo ""
echo "Deployment complete!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo "Your app is available at: ${SERVICE_URL}"

