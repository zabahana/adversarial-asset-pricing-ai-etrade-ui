# GitHub Repository Setup Guide

## 🚀 Quick Start

### Step 1: Prepare the Repository

Run the setup script:
```bash
./setup_git_repo.sh
```

This will:
- ✅ Add all files to Git
- ✅ Create an initial commit
- ✅ Show you the next steps

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `asset-pricing-ai-streamlit` (or your choice)
3. Description: "ARRL Multi-Head Attention Asset Pricing Agent"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

### Step 3: Push to GitHub

After creating the repo, GitHub will show you commands. Use these:

```bash
# Add the remote (replace USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Using SSH?** Replace the remote URL with:
```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
```

### Step 4: Verify

1. Go to your GitHub repository page
2. Verify all files are there
3. Check that `.gitignore` is working (shouldn't see `venv/`, `results/`, etc.)

---

## 🌐 Deploy to Streamlit Cloud

After pushing to GitHub:

### Option 1: Streamlit Cloud (Recommended)

1. **Sign up/Login**: https://streamlit.io/cloud
   - Use your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Python version: `3.11`

3. **Configure Secrets**:
   - Go to App Settings → Secrets
   - Add:
     ```
     ALPHA_VANTAGE_API_KEY = "your-api-key"
     OPENAI_API_KEY = "your-openai-key"  # Optional
     ```

4. **Deploy**:
   - Click "Deploy"
   - Wait for build to complete (~2-5 minutes)
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Option 2: Manual Deployment

See `DEPLOYMENT.md` for other deployment options (Cloud Run, Heroku, etc.)

---

## 📋 What Gets Committed

### ✅ Included:
- Source code (`streamlit_app.py`, `lightning_app/`, `src/`)
- Configuration files (`requirements.txt`, `.streamlit/config.toml`)
- Documentation (`README.md`, `DEPLOYMENT.md`)
- Setup scripts

### ❌ Excluded (via .gitignore):
- Virtual environment (`venv/`)
- Results and data files (`results/`, `data/`)
- Model checkpoints (`models/*.ckpt`)
- Secrets and credentials
- Cache and temporary files

---

## 🔐 Security Checklist

Before pushing, verify:
- ✅ No API keys in code
- ✅ No `.env` files committed
- ✅ No `credentials.json` committed
- ✅ `.gitignore` is working
- ✅ Only use environment variables or Streamlit secrets

---

## 🐛 Troubleshooting

### "Repository not found"
- Check the remote URL: `git remote -v`
- Verify you have access to the repository
- Make sure you're pushing to the correct branch

### "Permission denied"
- For HTTPS: Use a Personal Access Token instead of password
- For SSH: Set up SSH keys in GitHub settings

### "Large files error"
- Check `.gitignore` is working
- Remove large files: `git rm --cached large_file.pkl`
- Use Git LFS for large models if needed

---

## 📚 Additional Resources

- [GitHub Docs](https://docs.github.com/)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)

---

## ✅ Post-Deployment Checklist

After deploying to Streamlit Cloud:
- [ ] App loads without errors
- [ ] API keys work (test with a ticker)
- [ ] UI displays correctly
- [ ] Analysis runs successfully
- [ ] Results display properly


