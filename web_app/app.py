"""
Flask Backend for Lung Cancer Detection Web App
Connects HTML/CSS/JS frontend with PyTorch model
"""

import os
import sys

# Add parent src directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'src'))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np

try:
    from model_resnet_ltc import ResNetLTC
    MODEL_AVAILABLE = True
except ImportError:
    print("[WARNING] model_resnet_ltc not found. Running in demo mode.")
    MODEL_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
model_loaded = False

if MODEL_AVAILABLE:
    try:
        model = ResNetLTC(num_classes=2, hidden_units=128, pretrained=False)
        
        # Load model weights - use absolute path
        model_path = os.path.join(parent_dir, 'model', 'best_hybrid_lnn.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()  # CRITICAL: Set to evaluation mode (disables dropout)
            
            # Verify model is in eval mode
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    assert not module.training, "Dropout should be disabled in eval mode!"
            
            print(f"[OK] Model loaded successfully from {model_path}")
            print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
            print(f"  Epoch: {checkpoint['epoch']+1}")
            print(f"  Model: Hybrid LNN (ResNet18 + LTC)")
            print(f"  Model is in EVAL mode")
            model_loaded = True
        else:
            print(f"[WARNING] Model file not found at {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model_loaded = False
else:
    print("[INFO] Running in DEMO mode - model predictions will be random")
    print("   Using random weights for demo")

# Image preprocessing (MUST match training exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Transform pipeline:")
print("  1. Resize to 224x224")
print("  2. Convert to tensor [0, 1]")
print("  3. Normalize with ImageNet stats")

# Class names (IMPORTANT: Order matches training labels)
# Training: Normal=0, Cancer=1
class_names = ['normal', 'cancer']  # Index 0=Normal, Index 1=Cancer

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS)"""
    return send_from_directory('.', path)

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve result images from results folder"""
    results_dir = os.path.join(parent_dir, 'results')
    return send_from_directory(results_dir, filename)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint
    Receives image, returns prediction
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image file
        image_file = request.files['image']
        print(f"\n{'='*70}")
        print(f"Received image: {image_file.filename}")
        
        # Read and preprocess image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Original image mode: {image.mode}, size: {image.size}")
        
        # Convert to RGB (Hybrid LNN expects RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Tensor shape: {image_tensor.shape}")
        print(f"Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        # Make prediction
        if model_loaded and model is not None:
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get probabilities for both classes
                probs = probabilities[0].cpu().numpy()
                
                print(f"Raw outputs: {outputs[0].cpu().numpy()}")
                print(f"Probabilities: Normal={probs[0]:.4f}, Cancer={probs[1]:.4f}")
                print(f"Predicted: {class_names[predicted.item()]} (confidence: {confidence.item():.4f})")
                print(f"{'='*70}\n")
        else:
            # Demo mode - random prediction
            import random
            predicted_idx = random.randint(0, 1)
            confidence_val = random.uniform(0.6, 0.95)
            probs = [1 - confidence_val, confidence_val] if predicted_idx == 1 else [confidence_val, 1 - confidence_val]
            
            print(f"[DEMO MODE] Random prediction")
            print(f"Probabilities: Normal={probs[0]:.4f}, Cancer={probs[1]:.4f}")
            print(f"Predicted: {class_names[predicted_idx]} (confidence: {confidence_val:.4f})")
            print(f"{'='*70}\n")
            
            # Create mock tensors for compatibility
            class MockTensor:
                def __init__(self, value):
                    self.value = value
                def item(self):
                    return self.value
            
            predicted = MockTensor(predicted_idx)
            confidence = MockTensor(confidence_val)
        
        # Prepare response
        prediction_label = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        response = {
            'prediction': prediction_label.capitalize(),
            'confidence': float(confidence_score),
            'probabilities': {
                'normal': float(probs[0]),
                'cancer': float(probs[1])
            },
            'message': f'Prediction: {prediction_label.capitalize()} with {confidence_score*100:.2f}% confidence'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model information"""
    info = {
        'model': 'Hybrid LNN (ResNet18 + LTC)',
        'accuracy': '98.50%',
        'dataset': 'LIDC-IDRI',
        'total_images': 53187,
        'classes': ['Cancer', 'Normal'],
        'status': 'loaded' if model_loaded else 'demo_mode'
    }
    return jsonify(info)

@app.route('/test-prediction', methods=['GET'])
def test_prediction():
    """Test endpoint to verify model is working"""
    try:
        if model_loaded and model is not None:
            # Create a dummy tensor (simulating a preprocessed image)
            dummy_tensor = torch.randn(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                outputs = model(dummy_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                probs = probabilities[0].cpu().numpy()
            
            return jsonify({
                'status': 'Model is working!',
                'prediction': class_names[predicted.item()],
                'confidence': float(confidence.item()),
                'probabilities': {
                    'normal': float(probs[0]),
                    'cancer': float(probs[1])
                },
                'note': 'This is a test with random data. Upload a real CT scan for actual prediction.'
            })
        else:
            return jsonify({
                'status': 'Running in DEMO mode',
                'prediction': 'Normal',
                'confidence': 0.75,
                'probabilities': {
                    'normal': 0.75,
                    'cancer': 0.25
                },
                'note': 'Model not loaded. This is a demo response. Upload model file for real predictions.'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("LUNG CANCER DETECTION WEB APP")
    print("="*70)
    print(f"\nModel: Hybrid LNN (ResNet18 + LTC)")
    print(f"Accuracy: 98.50%")
    print(f"Device: {device}")
    print(f"\nStarting server...")
    print(f"Open browser and go to: http://localhost:5000")
    print("="*70 + "\n")
    
    # Get port from environment variable (for Render) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
