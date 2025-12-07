# Streamlit Troubleshooting Guide

## ✅ No Quotas for Local Streamlit

**Streamlit running locally has NO quotas** - it's just a Python library that runs on your computer.

## 🚨 Common Issues & Solutions

### Issue 1: "Cannot connect to localhost:8501"
**Solution:**
```bash
# Kill existing processes
pkill -f streamlit

# Start fresh
cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui
source venv/bin/activate
streamlit run streamlit_app.py --server.port=8501
```

### Issue 2: "Port already in use"
**Solution:**
```bash
# Find and kill the process
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run streamlit_app.py --server.port=8502
```

### Issue 3: "Module not found" errors
**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate

# Install missing dependencies
pip install -r requirements.txt
```

### Issue 4: Browser shows blank page or error
**Solution:**
1. Check browser console (F12) for JavaScript errors
2. Clear browser cache
3. Try a different browser
4. Check terminal for Python errors

### Issue 5: Analysis doesn't complete
**Solution:**
1. Check terminal logs - they show exactly what's happening
2. Check if API keys are set (Alpha Vantage, etc.)
3. Check disk space: `df -h`
4. Check memory: `top` or Activity Monitor

## 🔧 Quick Diagnostics

```bash
# Check if Streamlit is running
curl http://localhost:8501/_stcore/health

# Check what's using port 8501
lsof -i:8501

# Check Streamlit version
streamlit --version

# Check Python version
python3 --version

# Check disk space
df -h

# Check memory (macOS)
vm_stat
```

## 📊 Resource Limits

**Local Streamlit uses YOUR computer's resources:**
- CPU: Limited by your processor
- RAM: Limited by your system memory
- Disk: Limited by your hard drive space
- Network: Limited by your internet connection

**These are NOT quotas** - they're your computer's physical limits.

## 🌐 Cloud Run (Different from Local)

If you're asking about Cloud Run deployment:
- **GPU Quota**: Check GCP Console → IAM & Admin → Quotas
- **Billing**: Check GCP Console → Billing
- **Resource Limits**: Set in deployment config

---

**Still having issues?** Describe the exact error message or behavior you're seeing.
