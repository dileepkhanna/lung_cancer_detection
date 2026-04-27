# 🫁 Lung Cancer Detection Web App

## Beautiful Web Interface for AI-Powered CT Scan Analysis

**Accuracy**: 99.25% | **Model**: ResNet18 | **Dataset**: LIDC-IDRI

---

## 📋 Features

✅ **Drag & Drop Upload** - Easy image upload interface
✅ **Real-time Prediction** - Instant AI analysis
✅ **Visual Results** - Beautiful result display with confidence scores
✅ **Responsive Design** - Works on desktop, tablet, and mobile
✅ **Professional UI** - Modern, clean interface
✅ **Confidence Scores** - Shows prediction confidence
✅ **Probability Breakdown** - Displays probabilities for both classes

---

## 🚀 Quick Start

### Option 1: Run with Backend (Full Functionality)

1. **Install Requirements**:
   ```bash
   pip install flask flask-cors torch torchvision pillow
   ```

2. **Start the Server**:
   ```bash
   cd web_app
   python app.py
   ```

3. **Open Browser**:
   ```
   http://localhost:5000
   ```

4. **Upload CT Scan** and get instant prediction!

---

### Option 2: Demo Mode (Frontend Only)

1. **Open HTML File**:
   ```bash
   cd web_app
   # Double-click index.html or
   start index.html  # Windows
   open index.html   # Mac
   ```

2. **Note**: Demo mode uses simulated predictions. For real predictions, use Option 1.

---

## 📁 File Structure

```
web_app/
├── index.html          # Main HTML page
├── style.css           # Styling (beautiful UI)
├── script.js           # Frontend logic
├── app.py              # Flask backend (connects to model)
└── README.md           # This file
```

---

## 🔧 How It Works

### Frontend (HTML/CSS/JS):
1. User uploads CT scan image
2. Image is displayed for preview
3. User clicks "Analyze Image"
4. Loading animation shows
5. Results displayed with confidence scores

### Backend (Python/Flask):
1. Receives uploaded image
2. Preprocesses image (resize, normalize)
3. Runs through trained ResNet18 model
4. Returns prediction with probabilities
5. Frontend displays results

---

## 🎨 UI Features

### Upload Section:
- Drag & drop support
- Click to browse
- Supported formats: PNG, JPG, JPEG, DICOM
- Hover effects and animations

### Results Display:
- ✅ Normal or ⚠️ Cancer badge
- Confidence score with progress bar
- Probability breakdown for both classes
- Medical disclaimer
- Professional color scheme

### Model Info:
- Accuracy: 99.25%
- Model: ResNet18
- Dataset: LIDC-IDRI
- Images: 53,187

---

## 🔌 API Endpoints

### POST /predict
**Upload image and get prediction**

**Request**:
```
POST http://localhost:5000/predict
Content-Type: multipart/form-data
Body: image file
```

**Response**:
```json
{
  "prediction": "Normal",
  "confidence": 0.9825,
  "probabilities": {
    "cancer": 0.0175,
    "normal": 0.9825
  },
  "message": "Prediction: Normal with 98.25% confidence"
}
```

### GET /model-info
**Get model information**

**Response**:
```json
{
  "model": "ResNet18",
  "accuracy": "99.25%",
  "dataset": "LIDC-IDRI",
  "total_images": 53187,
  "classes": ["Cancer", "Normal"],
  "status": "loaded"
}
```

---

## 🎯 Usage Example

1. **Start Server**:
   ```bash
   python app.py
   ```

2. **Open Browser**: `http://localhost:5000`

3. **Upload Image**:
   - Drag & drop CT scan image
   - Or click "Choose File"

4. **Get Results**:
   - Click "Analyze Image"
   - Wait 1-2 seconds
   - View prediction with confidence

5. **Analyze Another**:
   - Click "Analyze Another Image"
   - Upload new image

---

## 📊 Model Details

### Architecture:
- **Base**: ResNet18 (pretrained on ImageNet)
- **Input**: 128×128 grayscale images
- **Output**: 2 classes (Cancer, Normal)
- **Accuracy**: 99.25% on test set
- **AUC-ROC**: 0.9997

### Preprocessing:
1. Convert to grayscale
2. Resize to 128×128
3. Normalize to [-1, 1]
4. Apply lung windowing

---

## ⚠️ Important Notes

### Medical Disclaimer:
This is an AI prediction tool for **research and educational purposes only**. 
Always consult with qualified medical professionals for diagnosis and treatment.

### Model Path:
The backend expects the model at: `../models/best_model_simple_resnet.pth`

If model not found, it runs in demo mode with random predictions.

---

## 🛠️ Customization

### Change Colors:
Edit `style.css`:
```css
/* Primary gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Change to your colors */
background: linear-gradient(135deg, #your-color-1 0%, #your-color-2 100%);
```

### Change Model:
Edit `app.py`:
```python
# Change model path
model_path = '../models/your_model.pth'

# Change class names
class_names = ['your_class_1', 'your_class_2']
```

---

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution**: Check model path in `app.py`. Should point to your trained model.

### Issue: "CORS error"
**Solution**: Make sure Flask-CORS is installed: `pip install flask-cors`

### Issue: "Port already in use"
**Solution**: Change port in `app.py`: `app.run(port=5001)`

### Issue: "Prediction takes too long"
**Solution**: 
- Use GPU if available
- Reduce image size
- Check CPU usage

---

## 📱 Mobile Support

The web app is fully responsive and works on:
- ✅ Desktop (Windows, Mac, Linux)
- ✅ Tablets (iPad, Android tablets)
- ✅ Mobile phones (iOS, Android)

---

## 🚀 Deployment Options

### Local Development:
```bash
python app.py
```

### Production (Gunicorn):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## 📝 Requirements

```
flask>=2.0.0
flask-cors>=3.0.0
torch>=1.9.0
torchvision>=0.10.0
pillow>=8.0.0
numpy>=1.19.0
```

Install all:
```bash
pip install flask flask-cors torch torchvision pillow numpy
```

---

## 🎉 Demo

1. Start server: `python app.py`
2. Open: `http://localhost:5000`
3. Upload a CT scan image
4. Get instant prediction!

---

## 📞 Support

For issues or questions:
1. Check model path is correct
2. Verify all dependencies installed
3. Check console for error messages
4. Ensure model file exists

---

**Developed with ❤️ for Medical AI Research**
**Model Accuracy: 99.25% | Dataset: LIDC-IDRI | Images: 53,187**
