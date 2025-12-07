# Lightning AI Studio - Web Interface Setup Steps

**Project Name:** `adversarial-asset-pricing-ai-v2`

Follow these steps to create and run the web interface in Lightning AI Studio.

---

## Step 1: Navigate to Your Project Directory

In the Lightning Studio terminal, run:

```bash
cd /teamspace/studios/this_studio
ls -la
```

Find and navigate to your project folder (should be named `adversarial-asset-pricing-ai-v2`):

```bash
cd adversarial-asset-pricing-ai-v2
pwd  # Verify you're in the right directory
```

**Expected output:** `/teamspace/studios/this_studio/adversarial-asset-pricing-ai-v2`

---

## Step 2: Verify Project Structure

Check that all necessary files are present:

```bash
ls -la lightning_app/
ls -la lightning_app/config.py
ls -la lightning_app/app.py
ls -la lightning_app/flows/
ls -la lightning_app/works/
ls -la lightning_app/ui/
ls -la requirements.txt
```

You should see:
- ✅ `lightning_app/config.py` (with your Alpha Vantage API key)
- ✅ `lightning_app/app.py`
- ✅ `lightning_app/flows/orchestrator.py`
- ✅ `lightning_app/works/data_fetch_work.py`
- ✅ `lightning_app/works/feature_engineering_work.py`
- ✅ `lightning_app/works/model_inference_work.py`
- ✅ `lightning_app/works/sentiment_work.py`
- ✅ `lightning_app/works/macro_work.py`
- ✅ `lightning_app/ui/dashboards.py`
- ✅ `requirements.txt`

---

## Step 3: Verify API Key Configuration

Check that your Alpha Vantage API key is set in the config:

```bash
cat lightning_app/config.py
```

**Expected output:**
```python
ALPHA_VANTAGE_API_KEY = "5X8N02ORS7PVFFZ4"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
...
```

If the API key is missing or incorrect, edit the file:
```bash
nano lightning_app/config.py
```
Or use the file browser to edit it directly.

---

## Step 4: Check Model Checkpoints

Verify your model checkpoints exist:

```bash
ls -la models/
ls -la models/dqn/
ls -la models/mha_dqn/
```

**Expected files:**
- `models/dqn/latest.ckpt`
- `models/mha_dqn/clean.ckpt`
- `models/mha_dqn/adversarial.ckpt`

**⚠️ If models are missing:**
- You'll need to upload your trained model checkpoints
- Or create placeholder PyTorch models for testing
- The app will fail during inference if models are missing

---

## Step 5: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**This will install:**
- `lightning` (Lightning AI framework)
- `streamlit` (Web dashboard)
- `torch` (PyTorch for models)
- `pandas`, `numpy`, `plotly` (Data processing & visualization)
- `requests`, `alpha_vantage` (API calls)
- And all other dependencies

**Expected output:** Packages will be downloaded and installed. This may take 1-2 minutes.

**💡 If you have a virtual environment:**
```bash
# Activate venv if you have one
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 6: Test Configuration

Quick test to verify everything is set up correctly:

```bash
python3 -c "from lightning_app.config import ALPHA_VANTAGE_API_KEY; print('✓ API Key loaded:', ALPHA_VANTAGE_API_KEY[:5] + '...')"
```

**Expected output:** `✓ API Key loaded: 5X8N0...`

If you see an error, check that:
- You're in the correct directory
- `lightning_app/config.py` exists and has the API key

---

## Step 7: Test Imports

Verify that all Python modules can be imported:

```bash
python3 -c "
from lightning_app.app import main
from lightning_app.flows.orchestrator import OrchestratorFlow
from lightning_app.works.data_fetch_work import DataFetchWork
print('✓ All imports successful!')
"
```

**Expected output:** `✓ All imports successful!`

If you see import errors, make sure:
- All dependencies are installed (`pip install -r requirements.txt`)
- You're in the project root directory

---

## Step 8: Run the Lightning App

Launch the application:

```bash
lightning run app lightning_app/app.py
```

**What happens:**
1. Lightning will initialize the app
2. The orchestrator flow will start
3. Data fetching will begin for the default ticker (NVDA)
4. Features will be engineered
5. Model inference will run
6. Sentiment data will be fetched
7. Macro data will be gathered
8. The Streamlit dashboard will start

**First run may take 2-3 minutes** as it:
- Downloads 5 years of stock data from Alpha Vantage
- Processes features
- Runs model inference
- Fetches news sentiment
- Gathers macro indicators

---

## Step 9: Monitor the Terminal Output

Watch the terminal for progress messages:

```
✅ Data fetching complete
✅ Feature engineering complete
✅ Model inference complete
✅ Sentiment analysis complete
✅ Macro data gathered
🚀 Dashboard starting...
```

**If you see errors:**
- Check the error message
- Verify API key is correct
- Ensure model checkpoints exist
- Check network connectivity

---

## Step 10: Access the Web Interface

After the app starts, Lightning will provide a URL in the terminal output, like:

```
🌐 Your app is running at: http://localhost:8501
```

Or if deployed to Lightning Cloud:

```
🌐 Your app is running at: https://adversarial-asset-pricing-ai-v2.lightning.ai
```

**To access:**
1. **Click the URL** in the terminal (if clickable)
2. **Or copy and paste** the URL into your browser
3. The Streamlit dashboard will open

---

## Step 11: Explore the Dashboard

The web interface will display:

### 📈 **Price History Chart**
- 5-year historical price data for the ticker
- Interactive Plotly chart

### 📊 **Model Comparison Table**
Performance metrics for:
- **Baseline DQN** (standard deep Q-network)
- **MHA-DQN (Clean)** (multi-head attention DQN, non-robust)
- **MHA-DQN (Robust)** (multi-head attention DQN, adversarial robust)

Metrics shown:
- Sharpe Ratio
- CAGR (Compound Annual Growth Rate)
- Maximum Drawdown
- Robustness Score

### 📰 **News Sentiment Section**
- Overall sentiment score (average of recent articles)
- Recent news articles with:
  - Title
  - Sentiment label (Bullish/Bearish/Neutral)
  - Summary

### 🌍 **Macro Backdrop**
- CPI (Consumer Price Index)
- Real GDP growth
- Unemployment Rate
- (Note: If FRED API key is not provided, mock data will be shown)

---

## Step 12: Change the Ticker (Optional)

To analyze a different stock ticker:

### Option A: Edit Config File (Recommended)
1. **Stop the app** (Press `Ctrl+C` in terminal)
2. **Edit** `lightning_app/config.py`:
   ```bash
   nano lightning_app/config.py
   ```
3. **Change** the default ticker:
   ```python
   DEFAULT_TICKER = "AAPL"  # Change from "NVDA" to your desired ticker
   ```
4. **Save** the file (Ctrl+O, Enter, Ctrl+X in nano)
5. **Restart** the app:
   ```bash
   lightning run app lightning_app/app.py
   ```

### Option B: Edit Orchestrator Directly
1. **Stop the app** (Press `Ctrl+C`)
2. **Edit** `lightning_app/flows/orchestrator.py`:
   ```bash
   nano lightning_app/flows/orchestrator.py
   ```
3. **Change** the ticker:
   ```python
   self.ticker: str = "AAPL"  # Change from DEFAULT_TICKER
   ```
4. **Save** and **restart** the app

---

## Troubleshooting Common Issues

### ❌ **Error: "Missing ALPHA_VANTAGE_API_KEY"**
**Fix:**
```bash
cat lightning_app/config.py | grep ALPHA_VANTAGE_API_KEY
```
If not found, the config file may be missing or incomplete. Re-check Step 3.

### ❌ **Error: "FileNotFoundError: models/dqn/latest.ckpt"**
**Fix:** Upload model checkpoints:
```bash
# In Lightning Studio, use the file browser to upload:
# - models/dqn/latest.ckpt
# - models/mha_dqn/clean.ckpt
# - models/mha_dqn/adversarial.ckpt
```

### ❌ **Error: "ModuleNotFoundError: No module named 'streamlit'"**
**Fix:**
```bash
pip install -r requirements.txt
# Or specifically:
pip install streamlit==1.39.0
```

### ❌ **Error: "Connection timeout" or API errors**
**Possible causes:**
- Alpha Vantage API rate limit exceeded
- Invalid API key
- Network connectivity issues

**Fix:**
- Wait a few minutes and try again (rate limits)
- Verify API key in `lightning_app/config.py`
- Check Alpha Vantage API status

### ❌ **App hangs or doesn't start**
**Fix:**
1. Check terminal for error messages
2. Try restarting the Lightning Studio session
3. Verify all files are in the correct location
4. Check that port 8501 is not already in use

### ❌ **Dashboard shows "Loading..." or empty**
**Fix:**
- Wait for the first data fetch to complete (may take 1-2 minutes)
- Check terminal logs for errors
- Verify model checkpoints are valid PyTorch files

---

## Next Steps After Setup

Once everything is working:

1. **Customize the Dashboard**
   - Edit `lightning_app/ui/dashboards.py` to add more visualizations
   - Add more financial metrics
   - Enhance the UI layout

2. **Add More Models**
   - Extend `ModelInferenceWork` to include additional model variants
   - Compare more strategies

3. **Deploy to Lightning Cloud**
   ```bash
   lightning run cloud lightning_app/app.py --cloud
   ```
   This will deploy your app to Lightning Cloud for persistent access.

4. **Add Real-Time Updates**
   - Configure the orchestrator to refresh data periodically
   - Add scheduled re-runs for model inference

5. **Add Ticker Input Field**
   - Modify the dashboard to accept user input for ticker selection
   - Update the orchestrator to accept dynamic tickers

---

## Quick Reference Commands

```bash
# Navigate to project
cd /teamspace/studios/this_studio/adversarial-asset-pricing-ai-v2

# Install dependencies
pip install -r requirements.txt

# Run the app
lightning run app lightning_app/app.py

# Stop the app
Ctrl+C

# Check API key
cat lightning_app/config.py | grep ALPHA_VANTAGE_API_KEY

# Verify models
ls -la models/dqn/ models/mha_dqn/

# Test imports
python3 -c "from lightning_app.config import ALPHA_VANTAGE_API_KEY; print('OK')"
```

---

## Support

If you encounter issues not covered here:

1. **Check Lightning AI Documentation:** https://lightning.ai/docs
2. **Review Terminal Logs:** Look for specific error messages
3. **Verify File Locations:** Ensure all files are in the correct directories
4. **Check Dependencies:** Run `pip list` to verify packages are installed

---

**You're all set! 🚀**

Follow these steps to get your web interface running in Lightning AI Studio.

