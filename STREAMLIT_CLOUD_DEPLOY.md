# 🚀 BOMOCO - Streamlit Cloud Deployment Guide

Deploy your Business-Outcome-Aware Multi-Objective Cloud Optimizer to Streamlit Cloud in minutes!

---

## Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Streamlit Cloud Account** - Free at [share.streamlit.io](https://share.streamlit.io)
3. **API Credentials** (Optional) - For live data from AWS, WattTime, etc.

---

## Step 1: Push to GitHub

### Option A: Create New Repository

```bash
# Initialize git in your BOMOCO folder
cd bomoco
git init
git add .
git commit -m "Initial BOMOCO deployment"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/bomoco.git
git branch -M main
git push -u origin main
```

### Option B: Fork/Clone Existing

If BOMOCO is already on GitHub, fork it to your account.

---

## Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account (if not already)
4. Select:
   - **Repository:** `YOUR_USERNAME/bomoco`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**

The app will build and deploy in 2-5 minutes.

---

## Step 3: Configure Secrets (Optional)

For live data instead of demo mode:

1. Go to your app's **Settings** → **Secrets**
2. Paste your secrets in TOML format:

```toml
# AWS Credentials
[aws]
access_key_id = "AKIAXXXXXXXXXXXXXXXX"
secret_access_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
region = "us-east-1"

# WattTime Carbon API
[watttime]
username = "your_username"
password = "your_password"

# Electricity Maps API
[electricity_maps]
api_key = "your_api_key"

# App Settings
[app]
demo_mode = false
```

3. Click **"Save"** - App will restart automatically

---

## Project Structure for Streamlit Cloud

```
bomoco/
├── .streamlit/
│   └── config.toml          # Theme and settings
├── app.py                    # Main application (entry point)
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration constants
├── data/
│   └── sample_data.py        # Demo data generators
├── utils/
│   └── multi_objective.py    # Optimization engine
├── integrations/             # API integrations (optional)
├── actuation/                # Actuation engine (optional)
└── patterns/                 # Architecture patterns (optional)
```

---

## Required Files

### 1. `requirements.txt`
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scipy>=1.11.0
scikit-learn>=1.3.0
python-dateutil>=2.8.0
requests>=2.31.0
boto3>=1.28.0
typing-extensions>=4.8.0
```

### 2. `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#7c3aed"
backgroundColor = "#0a0a0f"
secondaryBackgroundColor = "#16213e"
textColor = "#e2e8f0"

[server]
headless = true
enableCORS = false
```

---

## Troubleshooting

### App won't start?
- Check **Manage app** → **Logs** for errors
- Ensure all imports are in `requirements.txt`
- Verify `app.py` is at the root level

### Import errors?
- Add missing packages to `requirements.txt`
- Use relative imports: `from .module import func`

### Secrets not working?
- Check TOML syntax (use online validator)
- Restart app after saving secrets
- Access secrets via `st.secrets["section"]["key"]`

### Memory errors?
- Reduce data size in generators
- Use `@st.cache_data` for expensive operations
- Streamlit Cloud free tier has 1GB RAM limit

---

## Custom Domain (Optional)

1. Go to app **Settings** → **General**
2. Add your custom domain
3. Configure DNS:
   ```
   CNAME your-app.yourdomain.com -> your-app.streamlit.app
   ```

---

## Live Data Integration

### Enable AWS Integration

1. Create IAM user with these permissions:
   - `ce:GetCostAndUsage`
   - `ce:GetCostForecast`
   - `ce:GetRightsizingRecommendation`
   - `ec2:DescribeInstances`
   - `cloudwatch:GetMetricStatistics`

2. Add credentials to Streamlit Secrets

3. Set `demo_mode = false` in secrets

### Enable Carbon APIs

**WattTime** (Recommended):
1. Sign up at [watttime.org](https://www.watttime.org/api-documentation/)
2. Add username/password to secrets

**Electricity Maps** (Alternative):
1. Get API key at [electricitymaps.com](https://www.electricitymaps.com/)
2. Add to secrets

---

## Performance Tips

1. **Use caching:**
   ```python
   @st.cache_data(ttl=300)  # Cache for 5 minutes
   def load_data():
       return expensive_operation()
   ```

2. **Lazy loading:**
   ```python
   if st.button("Load Data"):
       data = load_data()
   ```

3. **Reduce initial load:**
   - Start with 20-30 workloads instead of 50
   - Use session_state to persist data

---

## Sharing Your App

Your deployed app URL will be:
```
https://YOUR_APP_NAME.streamlit.app
```

Share this URL with anyone - no login required!

For private access:
1. Go to **Settings** → **Sharing**
2. Set visibility to **Private**
3. Add allowed email addresses

---

## Updating Your App

Any push to your GitHub main branch automatically redeploys:

```bash
git add .
git commit -m "Update feature X"
git push
```

Streamlit Cloud will rebuild and deploy within minutes.

---

## Support

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **BOMOCO Issues:** Create issue in your GitHub repo

---

## Quick Deploy Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` at root
- [ ] `app.py` at root (or specify path)
- [ ] `.streamlit/config.toml` for theming
- [ ] Connected to Streamlit Cloud
- [ ] Secrets configured (if using live data)
- [ ] App URL shared with team

---

**Happy Deploying! 🎉**

Your BOMOCO platform is now live on Streamlit Cloud!
