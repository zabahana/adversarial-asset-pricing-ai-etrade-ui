# Run Streamlit App Locally - Quick Start

## 🚀 Run Locally for Debugging

```bash
cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui

# Option 1: Use helper script
./run_local.sh

# Option 2: Direct command
streamlit run streamlit_app.py --server.port=8501

# Option 3: With custom port
PORT=8502 streamlit run streamlit_app.py --server.port=8502
```

Then open: **http://localhost:8501**

## ✅ Advantages

- ✅ Real-time UI updates (no container rebuild)
- ✅ Easy debugging with browser console
- ✅ Direct file access
- ✅ Fast iteration on fixes
- ✅ See Streamlit reruns immediately

## 📋 Prerequisites

Make sure you have:
- Python 3.11+ installed
- Virtual environment activated
- Dependencies installed: `pip install -r requirements.txt`

---

**This is the fastest way to debug the UI display issue!**


