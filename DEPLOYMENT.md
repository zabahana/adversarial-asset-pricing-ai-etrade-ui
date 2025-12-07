# Deployment Guide

## 🚀 Streamlit Cloud Deployment (Recommended)

Streamlit Cloud is the easiest way to deploy Streamlit apps directly from GitHub.

### Steps:

1. **Push to GitHub** (see below)

2. **Go to Streamlit Cloud**: https://streamlit.io/cloud

3. **Sign in** with your GitHub account

4. **Click "New app"**

5. **Configure**:
   - Repository: Select your GitHub repo
   - Branch: `main` (or `master`)
   - Main file path: `streamlit_app.py`
   - Python version: 3.11

6. **Add Secrets**:
   - Go to App Settings → Secrets
   - Add:
     ```
     ALPHA_VANTAGE_API_KEY = "your-key"
     OPENAI_API_KEY = "your-key"  # Optional
     ```

7. **Deploy!** Streamlit Cloud will automatically build and deploy your app.

### Requirements:
- ✅ `requirements.txt` in root directory
- ✅ `streamlit_app.py` as main file
- ✅ `.streamlit/config.toml` (optional, for theme)

---

## ☁️ Google Cloud Run Deployment

See `GCP_DEPLOYMENT_GUIDE.md` for detailed instructions.

Quick deploy:
```bash
./deploy_gcp.sh
```

---

## 📦 Requirements for Deployment

### Required Files:
- ✅ `streamlit_app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Git ignore patterns
- ✅ `README.md` - Documentation

### Optional Files:
- `.streamlit/config.toml` - Streamlit configuration
- `.streamlit/secrets.toml` - Local secrets (don't commit!)
- `Dockerfile` - For containerized deployment

### Environment Variables:
- `ALPHA_VANTAGE_API_KEY` - Required
- `OPENAI_API_KEY` - Optional

---

## 🔐 Security Notes

**Never commit:**
- API keys
- Credentials files
- `.env` files
- `secrets.toml` with real keys

Use environment variables or platform secrets management instead!


