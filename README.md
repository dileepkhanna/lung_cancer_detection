# Lung Cancer Detection Web Application

AI-powered lung cancer detection using Hybrid LNN (ResNet18 + LTC) with 98.50% accuracy.

## Features

- Upload lung CT scan images
- Real-time cancer detection
- Confidence scores and probabilities
- Clean, responsive web interface
- Hybrid neural network architecture

## Quick Start

### Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
cd web_app
python app.py
```

Open browser: http://localhost:5000

## Deployment

### Azure (Recommended)
See [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md) for detailed Azure deployment instructions.

**Quick Azure Deploy:**
1. Create Azure Web App (Python 3.10, Free F1 tier)
2. Set startup command: `bash startup.sh`
3. Deploy from GitHub or Local Git
4. Access at: `https://your-app-name.azurewebsites.net`

### Render
See [DEPLOYMENT.md](DEPLOYMENT.md) for Render deployment instructions.

## Project Structure

```
├── web_app/              # Flask web application
│   ├── app.py           # Backend API
│   ├── index.html       # Frontend UI
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
├── src/                 # Source code
│   ├── model_resnet_ltc.py
│   └── ...
├── model/               # Trained models (excluded from git)
├── startup.sh           # Azure startup script
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── AZURE_DEPLOYMENT.md # Azure deployment guide
```

## Technology Stack

- **Backend**: Flask, PyTorch
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Hybrid LNN (ResNet18 + LTC)
- **Deployment**: Azure App Service / Render

## Model Information

- **Architecture**: Hybrid LNN combining ResNet18 and Liquid Time-Constant Networks
- **Accuracy**: 98.50%
- **Dataset**: LIDC-IDRI lung CT scans
- **Input**: 224x224 RGB images

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Image prediction endpoint
- `GET /model-info` - Model information
- `GET /test-prediction` - Test endpoint

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
