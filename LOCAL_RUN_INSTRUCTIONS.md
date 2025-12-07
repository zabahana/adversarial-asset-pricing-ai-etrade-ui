# Running Streamlit App Locally

## 🚀 Quick Start

```bash
cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui
./run_local.sh
```

Or manually:

```bash
cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui
source venv/bin/activate
streamlit run streamlit_app.py --server.port=8501
```

## 🌐 Access

Open in browser: **http://localhost:8501**

## 📊 Viewing Logs

### Terminal Output
All logs will appear in the terminal where you ran the command:
- Training progress
- Model evaluation
- File operations
- Errors and warnings

### Browser Console
Press **F12** in browser to see:
- JavaScript errors
- Network requests
- Streamlit reruns

## 🔍 Debugging UI Display Issues

When running locally, you can:

1. **See real-time updates** - Streamlit reruns are visible immediately
2. **Check file paths** - Verify `results/amzn_model_results.json` exists
3. **Inspect session state** - Use browser console or add debug prints
4. **Test fixes instantly** - No container rebuild needed

## ✅ Expected Behavior

1. Analysis starts → Logs show progress
2. Training completes → Logs show "Training complete"
3. Results saved → Logs show file path
4. UI updates → Results display automatically

## 🐛 If Results Don't Display

1. Check terminal logs for errors
2. Open browser console (F12) for JavaScript errors
3. Look for "Debug Information" expander in UI
4. Verify results file exists: `results/amzn_model_results.json`

---

**The app is running in the background. Check your terminal for logs!**


