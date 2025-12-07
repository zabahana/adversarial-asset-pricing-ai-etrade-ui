# Lightning AI Studio - Step-by-Step Setup Guide

Follow these steps to create and run the web interface in Lightning AI Studio.

## Step 1: Navigate to Project Directory

In your Lightning Studio terminal, run:

```bash
cd /teamspace/studios/this_studio/adversarial-asset-pricing-ai
```

Or if the project is in a different location:

```bash
cd /teamspace/studios/this_studio
ls -la  # Find the adversarial-asset-pricing-ai folder
cd adversarial-asset-pricing-ai
```

## Step 2: Verify Project Structure

Check that the Lightning app files exist:

```bash
ls -la lightning_app/
ls -la lightning_app/works/
ls -la lightning_app/flows/
ls -la lightning_app/ui/
```

You should see:
- `lightning_app/app.py`
- `lightning_app/config.py` (with your API key)
- `lightning_app/flows/orchestrator.py`
- `lightning_app/works/` (data_fetch_work.py, feature_engineering_work.py, etc.)
- `lightning_app/ui/dashboards.py`

## Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Note:** If you're using a virtual environment, activate it first:
```bash
source venv/bin/activate  # if you have a venv
pip install -r requirements.txt
```

This will install:
- lightning (for the app framework)
- streamlit (for the web UI)
- torch (for model inference)
- pandas, numpy, plotly (for data processing and visualization)
- requests, alpha_vantage (for API calls)
- And all other dependencies

## Step 4: Verify Model Checkpoints

Check that your model checkpoints exist:

```bash
ls -la models/dqn/
ls -la models/mha_dqn/
```

You should have:
- `models/dqn/latest.ckpt`
- `models/mha_dqn/clean.ckpt`
- `models/mha_dqn/adversarial.ckpt`

**If models are missing:** The app will fail during inference. You can use placeholder PyTorch models for testing, or upload your trained checkpoints.

## Step 5: Verify API Key Configuration

Check that your Alpha Vantage API key is set in the config:

```bash
cat lightning_app/config.py | grep ALPHA_VANTAGE_API_KEY
```

You should see:
```python
ALPHA_VANTAGE_API_KEY = "5X8N02ORS7PVFFZ4"
```

## Step 6: Test the Configuration

Run a quick Python test to verify imports work:

```bash
python3 -c "from lightning_app.config import ALPHA_VANTAGE_API_KEY; print('API Key loaded:', ALPHA_VANTAGE_API_KEY[:5] + '...')"
```

If this works, your configuration is correct.

## Step 7: Run the Lightning App

Launch the Lightning application:

```bash
lightning run app lightning_app/app.py
```

**First Run:**
- Lightning will initialize the app
- It will download data for the default ticker (NVDA)
- It will process features, run model inference, fetch sentiment, and gather macro data
- This may take 1-2 minutes

**Expected Output:**
- You'll see logs showing each Work executing
- Data fetching progress
- Feature engineering progress
- Model inference progress
- Sentiment and macro data fetching
- A URL/link to the Streamlit dashboard will appear

## Step 8: Access the Web Interface

After the app starts, Lightning will provide a URL like:

```
http://localhost:XXXXX
```

Or if deployed to Lightning Cloud:

```
https://your-app-name.lightning.ai
```

**Click the URL** or copy it to your browser to open the Streamlit dashboard.

## Step 9: Using the Dashboard

The dashboard will show:

1. **Price History Chart** - 5-year historical price data
2. **Model Comparison Table** - Metrics for:
   - Baseline DQN
   - MHA-DQN (clean)
   - MHA-DQN (adversarial robust)
3. **News Sentiment** - Recent news articles and sentiment scores
4. **Macro Backdrop** - Macroeconomic indicators (or mock data if FRED API key not provided)

## Step 10: Change the Ticker (Optional)

To analyze a different stock:

1. **Stop the app** (Ctrl+C in terminal)
2. **Edit** `lightning_app/config.py`:
   ```python
   DEFAULT_TICKER = "AAPL"  # Change from "NVDA" to your desired ticker
   ```
3. **Or edit** `lightning_app/flows/orchestrator.py`:
   ```python
   self.ticker: str = "AAPL"  # Change from DEFAULT_TICKER
   ```
4. **Restart** the app:
   ```bash
   lightning run app lightning_app/app.py
   ```

## Troubleshooting

### Error: "Missing ALPHA_VANTAGE_API_KEY"
- **Fix:** Check `lightning_app/config.py` has the correct API key

### Error: "FileNotFoundError: models/dqn/latest.ckpt"
- **Fix:** Upload model checkpoints to the `models/` directory, or create placeholder models

### Error: Module not found (e.g., "No module named 'streamlit'")
- **Fix:** Run `pip install -r requirements.txt` again

### App hangs or doesn't start
- **Check:** Terminal output for error messages
- **Try:** Restart the Lightning Studio session
- **Verify:** All files are in the correct location

### Dashboard shows empty data
- **Check:** Terminal logs to see if data fetching completed
- **Verify:** Alpha Vantage API key is valid and has sufficient quota
- **Try:** Wait a minute for the first data fetch to complete

## Next Steps

Once everything is working:

1. **Customize the Dashboard** - Edit `lightning_app/ui/dashboards.py` to add more visualizations
2. **Add More Models** - Extend `ModelInferenceWork` to include additional model variants
3. **Deploy to Cloud** - Use `lightning run cloud lightning_app/app.py --cloud` to deploy
4. **Add Real-Time Updates** - Configure the orchestrator to refresh data periodically

## Support

If you encounter issues:
1. Check the Lightning AI documentation: https://lightning.ai/docs
2. Review terminal error messages
3. Verify all dependencies are installed correctly
4. Check that model checkpoints are valid PyTorch models

