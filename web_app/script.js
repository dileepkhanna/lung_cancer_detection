// Show selected filename
document.getElementById('fileInput').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Choose a file...';
    document.getElementById('fileName').textContent = fileName;
});

// Handle form submission
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }
    
    // Show loading
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('image', file);
        
        // Call backend API
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Hide loading, show results
        document.getElementById('loading').style.display = 'none';
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        alert('Error: ' + error.message + '\n\nPlease make sure:\n1. The server is running\n2. You uploaded a valid image\n3. The image is a CT scan');
        resetForm();
    }
});

// Display results
function displayResults(data) {
    const predictionValue = document.getElementById('predictionValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const cancerProb = document.getElementById('cancerProb');
    const normalProb = document.getElementById('normalProb');
    
    // Set prediction
    predictionValue.textContent = data.prediction;
    predictionValue.className = 'prediction-value ' + data.prediction.toLowerCase();
    
    // Set confidence
    const confidence = (data.confidence * 100).toFixed(2);
    confidenceValue.textContent = confidence + '%';
    
    // Set probabilities
    cancerProb.textContent = (data.probabilities.cancer * 100).toFixed(2) + '%';
    normalProb.textContent = (data.probabilities.normal * 100).toFixed(2) + '%';
    
    // Show results
    document.getElementById('results').style.display = 'block';
}

// Reset form
function resetForm() {
    document.getElementById('uploadForm').reset();
    document.getElementById('fileName').textContent = 'Choose a file...';
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'none';
}
