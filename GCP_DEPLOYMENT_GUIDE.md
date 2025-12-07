# GCP Cloud Run Deployment Guide

This guide shows you how to deploy your Streamlit app to Google Cloud Platform (GCP) using Cloud Run.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed locally
3. **Docker** (optional, Cloud Build will handle it)
4. **GCP Project** created

## Option 1: Deploy to Cloud Run (Recommended)

### Step 1: Install Google Cloud SDK

If not already installed:

```bash
# macOS
brew install --cask google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Verify installation
gcloud --version
```

### Step 2: Authenticate and Set Project

```bash
# Login to GCP
gcloud auth login

# Create or select a project
gcloud projects create adversarial-asset-pricing-ai --set-as-default
# OR select existing project
gcloud config set project YOUR_PROJECT_ID

# Get your project ID
gcloud config get-value project
```

### Step 3: Enable Required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 4: Set Environment Variables

```bash
export GCP_PROJECT_ID=$(gcloud config get-value project)
export GCP_REGION="us-central1"  # or your preferred region
```

### Step 5: Build and Deploy

**Option A: Using the deployment script**

```bash
chmod +x deploy_gcp.sh
./deploy_gcp.sh
```

**Option B: Manual deployment**

```bash
# Build and push Docker image
gcloud builds submit --tag gcr.io/${GCP_PROJECT_ID}/adversarial-asset-pricing-ai

# Deploy to Cloud Run
gcloud run deploy adversarial-asset-pricing-ai \
  --image gcr.io/${GCP_PROJECT_ID}/adversarial-asset-pricing-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --port 8501 \
  --set-env-vars ALPHA_VANTAGE_API_KEY=5X8N02ORS7PVFFZ4
```

### Step 6: Access Your App

After deployment, Cloud Run will provide a URL like:
```
https://adversarial-asset-pricing-ai-XXXXX-uc.a.run.app
```

## Option 2: Deploy to App Engine (Alternative)

### Step 1: Create app.yaml

```yaml
runtime: python311

env: standard

instance_class: F2

handlers:
- url: /.*
  script: streamlit_app.py

env_variables:
  ALPHA_VANTAGE_API_KEY: '5X8N02ORS7PVFFZ4'
```

### Step 2: Create requirements-appengine.txt

```txt
streamlit==1.39.0
pandas==2.2.3
numpy==2.1.2
plotly==5.24.1
requests==2.32.3
torch==2.8.0
alpha_vantage==3.0.0
```

### Step 3: Deploy

```bash
gcloud app deploy app.yaml
```

## Option 3: Deploy to Compute Engine VM

### Step 1: Create VM

```bash
gcloud compute instances create streamlit-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB
```

### Step 2: SSH and Setup

```bash
gcloud compute ssh streamlit-vm --zone=us-central1-a

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Clone repo and setup
git clone YOUR_REPO_URL
cd adversarial-asset-pricing-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt streamlit

# Run Streamlit
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### Step 3: Configure Firewall

```bash
gcloud compute firewall-rules create allow-streamlit \
  --allow tcp:8501 \
  --source-ranges 0.0.0.0/0 \
  --description "Allow Streamlit access"
```

## Option 4: Use Vertex AI Workbench (For Development)

### Step 1: Create Workbench Instance

```bash
gcloud notebooks instances create streamlit-notebook \
  --vm-image-project=deeplearning-platform-release \
  --vm-image-family=pytorch-2-0-cu118-notebooks \
  --machine-type=n1-standard-4 \
  --location=us-central1-a
```

### Step 2: Access and Run

1. Go to Vertex AI Workbench in GCP Console
2. Click "Open JupyterLab"
3. Upload your files
4. Run: `streamlit run streamlit_app.py`
5. Use port forwarding to access locally

## Environment Variables in GCP

### Cloud Run: Set via --set-env-vars

```bash
gcloud run deploy SERVICE_NAME \
  --set-env-vars ALPHA_VANTAGE_API_KEY=5X8N02ORS7PVFFZ4,FRED_API_KEY=YOUR_KEY
```

### App Engine: Set in app.yaml

```yaml
env_variables:
  ALPHA_VANTAGE_API_KEY: '5X8N02ORS7PVFFZ4'
```

### Compute Engine: Set in startup script

```bash
export ALPHA_VANTAGE_API_KEY=5X8N02ORS7PVFFZ4
```

## Storing Model Checkpoints in GCS

### Step 1: Create GCS Bucket

```bash
gsutil mb -p ${GCP_PROJECT_ID} -l us-central1 gs://${GCP_PROJECT_ID}-models
```

### Step 2: Upload Models

```bash
gsutil -m cp -r models/* gs://${GCP_PROJECT_ID}-models/
```

### Step 3: Download in App

Modify `streamlit_app.py` to download from GCS:

```python
from google.cloud import storage

def download_models_from_gcs(bucket_name, models_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix='models/'):
        blob.download_to_filename(f"{models_dir}/{blob.name}")
```

## Cost Estimation

- **Cloud Run**: Pay per request (~$0.40 per million requests)
- **App Engine**: Pay per instance (~$30-50/month for F2)
- **Compute Engine**: Pay per VM (~$50-100/month for e2-medium)
- **Vertex AI Workbench**: Pay per hour (~$0.40/hour for n1-standard-4)

## Monitoring and Logging

### View Logs

```bash
# Cloud Run logs
gcloud run services logs read adversarial-asset-pricing-ai --region us-central1

# App Engine logs
gcloud app logs tail -s default
```

### Set Up Monitoring

1. Go to Cloud Console → Monitoring
2. Create uptime check for your service URL
3. Set up alerts for errors

## Troubleshooting

### Port Issues
- Ensure port 8501 is exposed
- Check firewall rules
- Verify health check endpoint

### Memory Issues
- Increase Cloud Run memory: `--memory 4Gi`
- Optimize model loading (use smaller models)

### Timeout Issues
- Increase timeout: `--timeout 600`
- Optimize data fetching (add caching)

## Security Best Practices

1. **Use Secret Manager for API Keys**:
```bash
# Create secret
echo -n "5X8N02ORS7PVFFZ4" | gcloud secrets create alpha-vantage-key --data-file=-

# Grant access
gcloud secrets add-iam-policy-binding alpha-vantage-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

2. **Enable IAM authentication** (remove `--allow-unauthenticated`)

3. **Use HTTPS only** (Cloud Run provides this automatically)

## Next Steps

After deployment:
1. Test the public URL
2. Set up custom domain (optional)
3. Configure auto-scaling
4. Set up monitoring and alerts
5. Add CI/CD pipeline for automatic deployments

---

**Recommended Approach**: Use **Cloud Run** for the easiest deployment and cost-effective scaling.

