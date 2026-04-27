# Quick Start Guide

## 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Run Web Application

```bash
cd web_app
python app.py
```

Open browser: http://localhost:5000

## 3. Test the Model

1. Upload a lung CT scan image
2. Click "Analyze Image"
3. View prediction and confidence score

## 4. View Results

All training results and visualizations are in the `results/` folder:
- Accuracy graphs
- Confusion matrix
- ROC curve
- Detailed metrics (CSV files)

## 5. Model Information

- **Model File**: `model/best_hybrid_lnn.pth`
- **Architecture**: Hybrid LNN (ResNet18 + LTC)
- **Accuracy**: 98.50%
- **Input Size**: 224x224 RGB

## Troubleshooting

**Issue**: Model not loading
- **Solution**: Ensure `model/best_hybrid_lnn.pth` exists

**Issue**: Import errors
- **Solution**: Install all requirements: `pip install -r requirements.txt`

**Issue**: CUDA errors
- **Solution**: Model will automatically use CPU if CUDA unavailable

## Support

Check the main README.md for detailed documentation.
