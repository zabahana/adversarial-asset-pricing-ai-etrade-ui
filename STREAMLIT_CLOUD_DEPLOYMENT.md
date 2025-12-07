# 🚀 Streamlit Cloud Deployment Guide

This guide walks you through deploying your MHA-DQN Portfolio Optimization app to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: Your code must be pushed to GitHub
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io) (free)

## 🔑 Step 1: Push Code to GitHub

### If you haven't pushed yet:

```bash
# Make sure you're in the project directory
cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui

# Run the setup script
chmod +x push_to_github.sh
./push_to_github.sh
```

Or manually:

```bash
# Initialize git (if not already done)
git init

# Add all files
git add -A

# Commit
git commit -m "Initial commit: MHA-DQN Portfolio Optimization"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## 🌐 Step 2: Deploy to Streamlit Cloud

### Option A: Deploy via Web Interface (Recommended)

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)

2. **Sign in with GitHub**: Click "Sign in" and authorize Streamlit Cloud to access your GitHub repositories

3. **Create New App**:
   - Click "New app" button
   - Select your GitHub repository
   - Choose the branch (usually `main`)
   - Set the main file path: `streamlit_app.py`
   - Click "Deploy!"

4. **Configure Secrets**:
   - In the app settings, go to "Secrets"
   - Add your API keys:
   ```toml
   ALPHA_VANTAGE_API_KEY = "your-alpha-vantage-api-key"
   OPENAI_API_KEY = "your-openai-api-key"
   ```

5. **Advanced Settings** (Optional):
   - **Python version**: Python 3.11
   - **Memory**: 2GB (default should work)
   - **Timeout**: 120 seconds

### Option B: Deploy via Streamlit CLI

```bash
# Install Streamlit Cloud CLI (if needed)
pip install streamlit

# Login to Streamlit Cloud
streamlit login

# Deploy
streamlit deploy
```

## 🔧 Configuration Files

### `.streamlit/config.toml` (Already included)
```toml
[theme]
primaryColor = "#1e40af"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1e293b"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### `.streamlit/secrets.toml` (Set in Streamlit Cloud dashboard)
```toml
ALPHA_VANTAGE_API_KEY = "your-key-here"
OPENAI_API_KEY = "your-key-here"
```

**Note**: Never commit `secrets.toml` to git! It's already in `.gitignore`.

## 📦 Dependencies

The `requirements.txt` file is automatically used by Streamlit Cloud. It includes:

- `streamlit==1.39.0`
- `torch==2.8.0`
- `pandas==2.2.3`
- `plotly==5.24.1`
- `alpha_vantage==3.0.0`
- And other dependencies...

## ✅ Post-Deployment Checklist

- [ ] App is accessible at `https://your-app-name.streamlit.app`
- [ ] API keys are set in Streamlit Cloud secrets
- [ ] Test the app with a stock ticker (e.g., "NVDA", "AAPL")
- [ ] Verify that data fetching works
- [ ] Check that model training/inference runs correctly
- [ ] Monitor logs for any errors

## 🐛 Troubleshooting

### App Won't Start

1. **Check Logs**: In Streamlit Cloud dashboard, click on your app → "Manage app" → "Logs"
2. **Common Issues**:
   - Missing dependencies in `requirements.txt`
   - API keys not set correctly
   - Port conflicts (should use default 8501)
   - Memory issues (may need to optimize model loading)

### Import Errors

```bash
# Make sure all dependencies are in requirements.txt
pip freeze > requirements_check.txt
# Compare with requirements.txt
```

### Memory Issues

- Streamlit Cloud free tier: 1GB RAM
- Upgrade to Team plan for more resources
- Or optimize model loading (lazy loading, smaller models)

### Timeout Issues

- Default timeout: 120 seconds
- For long-running operations, consider:
  - Using async processing
  - Caching results
  - Background tasks

## 🔄 Updating Your Deployment

1. **Push Changes to GitHub**:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push
   ```

2. **Streamlit Cloud Auto-Deploys**: Changes are automatically detected and deployed

3. **Manual Redeploy**: In Streamlit Cloud dashboard → "Manage app" → "Reboot app"

## 📊 Monitoring

- **App Status**: Check dashboard for deployment status
- **Usage Stats**: View in Streamlit Cloud dashboard
- **Logs**: Access real-time logs in the dashboard

## 🔐 Security Best Practices

1. ✅ Never commit API keys or secrets
2. ✅ Use Streamlit Cloud secrets for sensitive data
3. ✅ Enable XSRF protection (already enabled in config)
4. ✅ Keep dependencies updated
5. ✅ Regularly review access logs

## 💰 Pricing

- **Free Tier**: Unlimited public apps, 1GB RAM
- **Team Plan**: Private apps, more resources, team collaboration
- **Enterprise**: Custom deployment, dedicated resources

## 📚 Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)

## 🆘 Support

- Streamlit Community Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: Open an issue in your repository
- Streamlit Support: Available for Team/Enterprise plans

---

**Your app URL will be**: `https://YOUR-APP-NAME.streamlit.app`

Happy deploying! 🎉

