#!/bin/bash
# Run Streamlit app locally for debugging

set -e

echo "🚀 Starting Streamlit App Locally"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found"
    echo "   Please run from adversarial-asset-pricing-ai-etrade-ui directory"
    exit 1
fi

# Set port
PORT=${PORT:-8501}

echo "📋 Configuration:"
echo "   • Port: $PORT"
echo "   • URL: http://localhost:$PORT"
echo ""

# Check for virtual environment and activate if available
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
fi

# Check dependencies
if ! command -v streamlit &> /dev/null && ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Error: streamlit not found"
    echo "   Install with: pip install streamlit"
    exit 1
fi

echo "✅ Starting Streamlit server..."
echo ""
echo "💡 Tips:"
echo "   • Press Ctrl+C to stop"
echo "   • Check browser console (F12) for errors"
echo "   • Results update in real-time"
echo "   • Logs will show below"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Use python -m streamlit to ensure we use the right environment
python3 -m streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=localhost \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --logger.level=info
