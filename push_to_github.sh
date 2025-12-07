#!/bin/bash
set -e

echo "🚀 Setting up GitHub Repository and Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Git not initialized. Running git init..."
    git init
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "📝 Staging all files..."
    git add -A
    
    echo "📦 Creating initial commit..."
    read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
    COMMIT_MSG=${COMMIT_MSG:-"Initial commit: MHA-DQN Portfolio Optimization with Streamlit UI"}
    
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✅ Commit created!${NC}"
else
    echo -e "${YELLOW}⚠️  No changes to commit${NC}"
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")

# Check if remote exists
if git remote get-url origin &>/dev/null; then
    REMOTE_URL=$(git remote get-url origin)
    echo -e "${BLUE}📍 Remote 'origin' already exists:${NC} $REMOTE_URL"
    read -p "Do you want to update it? (y/n): " UPDATE_REMOTE
    if [[ $UPDATE_REMOTE == "y" ]]; then
        read -p "Enter new GitHub repository URL: " NEW_REMOTE
        git remote set-url origin "$NEW_REMOTE"
    fi
else
    echo ""
    echo -e "${YELLOW}⚠️  No remote repository configured${NC}"
    echo ""
    echo "To connect to GitHub:"
    echo "1. Create a new repository on GitHub (https://github.com/new)"
    echo "2. Copy the repository URL (e.g., https://github.com/username/repo-name.git)"
    echo ""
    read -p "Enter your GitHub repository URL (or press Enter to skip): " REPO_URL
    
    if [ -n "$REPO_URL" ]; then
        git remote add origin "$REPO_URL"
        echo -e "${GREEN}✅ Remote 'origin' added!${NC}"
        
        echo ""
        echo "📤 Pushing to GitHub..."
        git push -u origin "$CURRENT_BRANCH"
        echo -e "${GREEN}✅ Pushed to GitHub!${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipping remote setup. You can add it later with:${NC}"
        echo "   git remote add origin <your-repo-url>"
        echo "   git push -u origin $CURRENT_BRANCH"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ GitHub setup complete!${NC}"
echo ""
echo "📋 Next Steps:"
echo "1. If not already done, create a repository on GitHub"
echo "2. Add remote: git remote add origin <your-repo-url>"
echo "3. Push: git push -u origin $CURRENT_BRANCH"
echo "4. Deploy to Streamlit Cloud: https://share.streamlit.io"
echo ""
echo "📖 See STREAMLIT_CLOUD_DEPLOYMENT.md for detailed deployment instructions"

