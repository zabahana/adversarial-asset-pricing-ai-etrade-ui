#!/bin/bash
# Setup Git repository and push to GitHub

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Setting up Git repository for GitHub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$(dirname "$0")"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if .git exists
if [ ! -d .git ]; then
    echo "📦 Initializing Git repository..."
    git init
else
    echo "✅ Git repository already initialized"
fi

# Add all files
echo ""
echo "📝 Adding files to Git..."
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "⚠️  No changes to commit"
else
    # Commit changes
    echo ""
    echo "💾 Committing changes..."
    git commit -m "Initial commit: ARRL Asset Pricing Agent with Streamlit UI

- Multi-Head Attention DQN implementation
- Streamlit web interface
- Adversarial robustness evaluation
- Real-time market data integration
- Comprehensive analysis and visualization"
    
    echo "✅ Changes committed"
fi

# Check for remote
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$REMOTE_URL" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📡 No GitHub remote configured"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "To push to GitHub:"
    echo ""
    echo "1. Create a new repository on GitHub:"
    echo "   https://github.com/new"
    echo ""
    echo "2. Run these commands (replace USERNAME and REPO_NAME):"
    echo ""
    echo "   git remote add origin https://github.com/USERNAME/REPO_NAME.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "OR use SSH:"
    echo ""
    echo "   git remote add origin git@github.com:USERNAME/REPO_NAME.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
else
    echo ""
    echo "✅ Remote configured: $REMOTE_URL"
    echo ""
    read -p "Push to GitHub? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Ensure we're on main branch
        CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
        if [ "$CURRENT_BRANCH" != "main" ]; then
            git branch -M main 2>/dev/null || git checkout -b main
        fi
        
        echo ""
        echo "📤 Pushing to GitHub..."
        git push -u origin main
        echo ""
        echo "✅ Successfully pushed to GitHub!"
        echo ""
        echo "🌐 Your repository: $REMOTE_URL"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. If not already done, create a GitHub repo and add remote"
echo "2. Push your code: git push -u origin main"
echo "3. Deploy to Streamlit Cloud:"
echo "   https://streamlit.io/cloud"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Select your repository"
echo "   - Main file: streamlit_app.py"
echo "   - Add secrets for API keys"
echo ""
echo "See DEPLOYMENT.md for detailed instructions."
echo ""


