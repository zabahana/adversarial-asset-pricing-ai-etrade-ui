# Quick Start: Run Locally with Logs

## 🚀 Start Streamlit App Locally

Run this in your terminal to see all logs in real-time:

```bash
cd /Users/zelalemabahana/adversarial-asset-pricing-ai-etrade-ui
source venv/bin/activate
streamlit run streamlit_app.py --server.port=8501
```

## 📊 What You'll See

The terminal will show:
- ✅ Streamlit server starting
- ✅ All analysis progress logs
- ✅ Training progress (episodes, loss)
- ✅ Model evaluation results
- ✅ File save operations
- ✅ Errors and warnings

## 🌐 Access UI

After starting, open: **http://localhost:8501**

## 🔍 Benefits of Running Locally

1. **Real-time logs** - See everything as it happens
2. **Instant UI updates** - No waiting for container rebuilds
3. **Easy debugging** - Check files, session state directly
4. **Fast iteration** - Test fixes immediately

## 📋 Example Log Output

You'll see logs like:
```
[INFO] Episode 10/80, Loss: 0.7069
[INFO] Episode 20/80, Loss: 0.7047
[COMPLETE] Training complete!
[FORECAST] ✅ Forecast generated: BUY with +0.38% change
[COMPLETE] Model evaluation complete. Results saved to: results/amzn_model_results.json
```

---

**Run the command above to start!**


