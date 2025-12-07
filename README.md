# ARRL Multi-Head Attention Asset Pricing Agent

Enterprise Agentic AI System for Asset Pricing with Multi-Head Attention Deep Q-Network (MHA-DQN).

## 🚀 Features

- **Multi-Head Attention DQN**: Advanced reinforcement learning agent for asset pricing
- **Adversarial Robustness**: Built-in adversarial training and evaluation
- **Real-time Data**: Fetches market data, sentiment, and fundamental analysis
- **Interactive UI**: Modern Streamlit-based web interface
- **Comprehensive Analysis**: Price forecasting, risk assessment, and performance metrics

## 📋 Prerequisites

- Python 3.11+
- API Keys:
  - Alpha Vantage API key (for market data)
  - OpenAI API key (optional, for LLM summaries)

## 🛠️ Installation

### Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd adversarial-asset-pricing-ai-etrade-ui
```

Or if you haven't pushed to GitHub yet:
```bash
cd adversarial-asset-pricing-ai-etrade-ui
./push_to_github.sh  # Setup and push to GitHub
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export ALPHA_VANTAGE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"  # Optional
```

Or create a `.streamlit/secrets.toml` file:
```toml
ALPHA_VANTAGE_API_KEY = "your-api-key"
OPENAI_API_KEY = "your-openai-key"
```

## 🎯 Usage

### Local Development

Run the Streamlit app locally:
```bash
streamlit run streamlit_app.py
```

Or use the helper script:
```bash
./start_local.sh
```

The app will be available at `http://localhost:8501`

### Cloud Deployment

#### 🚀 Streamlit Cloud (Recommended - Free & Easy)

1. **Push to GitHub**: 
   ```bash
   ./push_to_github.sh
   ```
   Or manually push your code to GitHub.

2. **Deploy to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file: `streamlit_app.py`
   - Add your API keys in "Secrets"
   - Click "Deploy!"

📖 **Detailed Guide**: See [STREAMLIT_CLOUD_DEPLOYMENT.md](STREAMLIT_CLOUD_DEPLOYMENT.md)

#### ☁️ Google Cloud Run

See `GCP_DEPLOYMENT_GUIDE.md` for detailed instructions on deploying with GPU support.

#### Other Platforms

- **AWS EC2/ECS**: Use Dockerfile provided
- **Heroku**: Requires Procfile (not included, but can be added)

## 📊 Project Structure

```
.
├── streamlit_app.py          # Main Streamlit application
├── lightning_app/            # Core application logic
│   ├── works/               # Data processing and model works
│   ├── flows/               # Orchestration flows
│   └── config.py            # Configuration
├── src/                     # Source code
│   ├── models/             # Model definitions
│   ├── data_pipeline/      # Data processing
│   └── utils/              # Utilities
├── results/                 # Analysis results (gitignored)
├── models/                  # Trained models (gitignored)
└── requirements.txt         # Python dependencies
```

## 🔧 Configuration

### Alpha Vantage API

Get your free API key from: https://www.alphavantage.co/support/#api-key

Set it as an environment variable or in `.streamlit/secrets.toml`:
```toml
ALPHA_VANTAGE_API_KEY = "your-key-here"
```

### Model Parameters

Adjust in the Streamlit UI sidebar:
- **Episodes**: Training episodes (20-500)
- **Risk Level**: Low, Medium, High
- **Historical Years**: Data history (1-10 years)
- **Enable Sentiment**: Include sentiment analysis

## 📈 Features

- **Price Forecasting**: Multi-step ahead price predictions
- **Risk Analysis**: Sharpe ratio, maximum drawdown, volatility
- **Adversarial Robustness**: FGSM, PGD, C&W, BIM, DeepFool attack evaluation
- **Fundamental Analysis**: 10K summary, financial metrics
- **Sentiment Analysis**: News and social media sentiment
- **Performance Metrics**: Detailed portfolio performance visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Alpha Vantage for market data API
- PyTorch for deep learning framework
- Streamlit for UI framework

## 📧 Contact

For questions or issues, please open an issue on GitHub.


