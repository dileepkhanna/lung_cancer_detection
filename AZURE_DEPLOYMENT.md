# Azure Deployment Guide - Free Tier

## Prerequisites
- Azure account (free tier available)
- Git installed
- Azure CLI (optional but recommended)

## Method 1: Deploy via Azure Portal (Easiest)

### Step 1: Create Azure Web App

1. Go to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource"
3. Search for "Web App" and click "Create"
4. Configure:
   - **Subscription**: Your subscription
   - **Resource Group**: Create new (e.g., "lung-cancer-rg")
   - **Name**: lung-cancer-detection (must be globally unique)
   - **Publish**: Code
   - **Runtime stack**: Python 3.10
   - **Operating System**: Linux
   - **Region**: Choose nearest region
   - **Pricing Plan**: Free F1 (100% free)

5. Click "Review + Create" → "Create"

### Step 2: Configure Deployment

1. Go to your Web App resource
2. In left menu, click "Deployment Center"
3. Choose source:
   - **GitHub**: Connect your GitHub account and select repository
   - **Local Git**: Use Azure's Git repository
   - **External Git**: Use any Git URL

4. Click "Save"

### Step 3: Configure Application Settings

1. In left menu, click "Configuration"
2. Under "General settings":
   - **Startup Command**: `bash startup.sh`
   - **Stack**: Python 3.10

3. Click "Save"

### Step 4: Deploy

If using GitHub:
- Push code to GitHub
- Azure auto-deploys on push

If using Local Git:
```bash
# Get Git URL from Deployment Center
git remote add azure <your-azure-git-url>
git push azure main
```

### Step 5: Access Your App

Your app will be available at:
`https://lung-cancer-detection.azurewebsites.net`

## Method 2: Deploy via Azure CLI

### Install Azure CLI

**Windows:**
```bash
winget install Microsoft.AzureCLI
```

**Mac:**
```bash
brew install azure-cli
```

**Linux:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Deploy Commands

```bash
# Login to Azure
az login

# Create resource group
az group create --name lung-cancer-rg --location eastus

# Create App Service plan (Free tier)
az appservice plan create --name lung-cancer-plan --resource-group lung-cancer-rg --sku F1 --is-linux

# Create Web App
az webapp create --resource-group lung-cancer-rg --plan lung-cancer-plan --name lung-cancer-detection --runtime "PYTHON:3.10"

# Configure startup command
az webapp config set --resource-group lung-cancer-rg --name lung-cancer-detection --startup-file "bash startup.sh"

# Deploy from local Git
az webapp deployment source config-local-git --name lung-cancer-detection --resource-group lung-cancer-rg

# Get deployment URL
az webapp deployment list-publishing-credentials --name lung-cancer-detection --resource-group lung-cancer-rg --query scmUri --output tsv

# Add Azure remote and push
git remote add azure <deployment-url>
git push azure main
```

## Method 3: Deploy from GitHub Actions (CI/CD)

Azure can auto-generate a GitHub Actions workflow:

1. In Deployment Center, select GitHub
2. Azure creates `.github/workflows/main_lung-cancer-detection.yml`
3. Every push to main branch auto-deploys

## Configuration Files Explained

### `startup.sh`
- Bash script that runs on container startup
- Changes to web_app directory
- Starts gunicorn server

### `requirements.txt`
- Lists all Python dependencies
- Azure automatically runs `pip install -r requirements.txt`

### `.deployment`
- Tells Azure to build during deployment
- Ensures dependencies are installed

## Important Notes

### Free Tier Limitations
- **CPU**: 60 minutes/day compute time
- **Memory**: 1 GB RAM
- **Storage**: 1 GB
- **Custom domains**: Not available on free tier
- **Always On**: Not available (app sleeps after 20 min idle)

### Model File Issue
⚠️ The model file (`model/best_hybrid_lnn.pth`) is too large for Git.

**Solutions:**
1. **Azure Blob Storage** (Recommended):
   - Upload model to Azure Blob Storage
   - Download on app startup
   
2. **Use demo mode**: App runs without model (random predictions)

3. **Azure Files**: Mount persistent storage (paid feature)

### Environment Variables

Set in Azure Portal → Configuration → Application Settings:
- `PYTHON_VERSION=3.10`
- `SCM_DO_BUILD_DURING_DEPLOYMENT=true`
- `WEBSITES_PORT=8000`

## Monitoring and Logs

### View Logs
```bash
# Stream logs
az webapp log tail --name lung-cancer-detection --resource-group lung-cancer-rg

# Or in Azure Portal:
# Your Web App → Log stream
```

### Check App Status
```bash
az webapp show --name lung-cancer-detection --resource-group lung-cancer-rg --query state
```

## Troubleshooting

### App not starting
1. Check logs in Azure Portal → Log stream
2. Verify startup command: `bash startup.sh`
3. Ensure Python version is 3.10

### Build fails
1. Check `requirements.txt` is valid
2. Verify all dependencies are compatible
3. Check deployment logs

### App crashes
1. Check application logs
2. Verify gunicorn is installed
3. Test locally first: `bash startup.sh`

### Timeout errors
- Increased timeout to 600 seconds in startup.sh
- Free tier has limited resources

## Testing Locally

```bash
# Make startup script executable
chmod +x startup.sh

# Run startup script
bash startup.sh

# Or test directly
cd web_app
gunicorn --bind=0.0.0.0:8000 app:app
```

Open browser: http://localhost:8000

## Updating Your App

```bash
# Make changes
git add .
git commit -m "Update message"

# Push to Azure (auto-deploys)
git push azure main

# Or push to GitHub (if using GitHub deployment)
git push origin main
```

## Scaling (Paid Tiers)

To upgrade from free tier:
```bash
az appservice plan update --name lung-cancer-plan --resource-group lung-cancer-rg --sku B1
```

Pricing tiers:
- **F1**: Free (60 min/day)
- **B1**: Basic (~$13/month, always on)
- **S1**: Standard (~$70/month, auto-scale)

## Clean Up Resources

To delete everything and stop charges:
```bash
az group delete --name lung-cancer-rg --yes
```

## Useful Links

- [Azure Portal](https://portal.azure.com)
- [Azure App Service Docs](https://docs.microsoft.com/azure/app-service/)
- [Azure Free Tier](https://azure.microsoft.com/free/)
- [Python on Azure](https://docs.microsoft.com/azure/app-service/quickstart-python)

## Support

For issues:
1. Check Azure Portal logs
2. Review deployment logs
3. Test locally first
4. Check Azure status page

## Cost Estimate

**Free Tier**: $0/month
- 10 web apps
- 1 GB storage per app
- 60 CPU minutes/day per app

**Basic B1**: ~$13/month
- Always on
- Custom domains
- 1.75 GB RAM
- 10 GB storage
