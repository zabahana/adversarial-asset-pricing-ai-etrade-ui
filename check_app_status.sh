#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Streamlit App Status Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check local Streamlit process
echo "🔍 LOCAL STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if lsof -ti:8501 >/dev/null 2>&1; then
    PID=$(lsof -ti:8501)
    echo -e "${GREEN}✅ Streamlit is running locally${NC}"
    echo "   • Port: 8501"
    echo "   • Process ID: $PID"
    echo "   • URL: http://localhost:8501"
    
    # Check if it's responding
    if curl -s http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        echo -e "   • Health: ${GREEN}Healthy${NC}"
    else
        echo -e "   • Health: ${YELLOW}Not responding${NC}"
    fi
else
    echo -e "${RED}❌ Streamlit is NOT running locally${NC}"
    echo "   • Port 8501 is free"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "☁️  STREAMLIT CLOUD STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if GitHub repo exists
REPO_URL="https://github.com/zabahana/adversarial-asset-pricing-ai-etrade-ui"
if curl -s -o /dev/null -w "%{http_code}" "$REPO_URL" | grep -q "200"; then
    echo -e "${GREEN}✅ Repository is accessible${NC}"
    echo "   • URL: $REPO_URL"
else
    echo -e "${RED}❌ Repository not found${NC}"
fi

echo ""
echo "📋 To check Streamlit Cloud deployment:"
echo "   1. Visit: https://share.streamlit.io"
echo "   2. Sign in with GitHub"
echo "   3. Look for your app: adversarial-asset-pricing-ai-etrade-ui"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 QUICK ACTIONS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To start locally:"
echo "   ./start_fresh.sh"
echo ""
echo "To check Streamlit Cloud:"
echo "   Open: https://share.streamlit.io"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

