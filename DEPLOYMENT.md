# Deployment Guide - Render Free Tier

## Prerequisites
- Git installed
- GitHub account
- Render account (free)

## Step 1: Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Lung Cancer Detection App"
```

## Step 2: Push to GitHub

```bash
# Create a new repository on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: lung-cancer-detection (or your choice)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --chdir web_app app:app`
   - **Plan**: Free

5. Click "Create Web Service"

## Step 4: Wait for Deployment

- First deployment takes 5-10 minutes
- Render will install dependencies and start your app
- You'll get a URL like: `https://lung-cancer-detection.onrender.com`

## Important Notes

### Model File
⚠️ **The model file (`model/best_hybrid_lnn.pth`) is excluded from Git due to size.**

**Options:**
1. **Use demo mode**: App will work with random weights (for testing UI)
2. **Upload model manually**: Use Render's persistent disk (paid feature)
3. **Host model externally**: Store on Google Drive/Dropbox and download on startup

### Free Tier Limitations
- App sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds to wake up
- 750 hours/month free (enough for one service)
- No persistent storage (files reset on redeploy)

### Environment Variables (Optional)
If needed, add in Render dashboard:
- `PYTHON_VERSION=3.10.0`
- `PORT` (automatically set by Render)

## Testing Locally Before Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Run with gunicorn (production server)
gunicorn --chdir web_app app:app

# Or run with Flask dev server
cd web_app
python app.py
```

## Troubleshooting

### Build fails
- Check `requirements.txt` has all dependencies
- Verify Python version compatibility

### App crashes on startup
- Check Render logs in dashboard
- Ensure `gunicorn` is in requirements.txt
- Verify `PORT` environment variable is used

### Model not loading
- Expected behavior if model file not uploaded
- App will show warning but still run in demo mode

## Git Commands Reference

```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push

# Pull latest changes
git pull
```

## Updating Your Deployment

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update description"
   git push
   ```
3. Render auto-deploys from GitHub (if enabled)
4. Or manually trigger deploy in Render dashboard

## Support

- [Render Documentation](https://render.com/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)
