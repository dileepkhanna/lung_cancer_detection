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

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on deploying to Render.

### Quick Deploy to Render

1. Push code to GitHub
2. Connect repository to Render
3. Deploy automatically

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
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── DEPLOYMENT.md       # Deployment guide
```

## Technology Stack

- **Backend**: Flask, PyTorch
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Hybrid LNN (ResNet18 + LTC)
- **Deployment**: Render (free tier)

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
