from fastapi import FastAPI, UploadFile, File, HTTPException # fastapi>=0.103.1
from fastapi.responses import HTMLResponse
import torch # torch==2.2.0+cpu
import torch.nn as nn
from torchvision import transforms # torchvision==0.17.0+cpu
from PIL import Image # Pillow>=10.1.0
import io
import os
import logging
import sys
from prometheus_fastapi_instrumentator import Instrumentator # prometheus-fastapi-instrumentator>=6.1.0
import argparse

# Add project root to path to fix ModuleNotFoundError: No module named 'src'
# This ensures that imports from the 'src' package work correctly across different environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the SimpleCNN architecture - must match the architecture used during training
# Structure: 2 Convolutional layers, 2 Max-pooling layers, and 2 Fully-connected layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# OpenAPI Tags for metadata
tags_metadata = [
    {
        "name": "Team Members - Group 115",
        "description": """
1. MAJJIGI JAMBANNA (2024AA05721)
2. NIRANJAN KUMAR SHARMA (2024AA05405)
3. SRIDEVI THAKKU RAMANAN (2024AA05402)
4. TRIPTI (2024AA05593)
5. HARSHAL KISHORE PHADAS (2024AA05139)
""",
    }
]

app = FastAPI(
    title="Group 115: MLOps Cats vs Dogs API",
    description="A simple API to classify images of cats and dogs using a trained CNN model. Team: Group 115",
    version="1.0.0",
    openapi_tags=tags_metadata
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Device configuration: use GPU (cuda) if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the model weights from the 'models/' directory
# Supports dynamic loading via the --model_name CLI argument
def load_model(model_name="model.pt"):
    global MODEL_PATH, model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", model_name)
    model = SimpleCNN().to(device)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            logger.info(f"Model {model_name} loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            return False
    else:
        logger.warning(f"Model file {MODEL_PATH} not found. Prediction will fail.")
        return False

# Default load for production/docker (loads model.pt if no args provided)
if __name__ == "src.app" or __name__ == "app":
    load_model()

# Standard Image Transformations for Inference:
# 1. Resize to (224, 224) as required by the model
# 2. Convert to Tensor
# 3. Normalize using standard ImageNet mean/standard deviation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Group 115: Cats vs Dogs Classifier</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #4F46E5;
                --primary-hover: #4338CA;
                --bg: #F9FAFB;
                --card-bg: #FFFFFF;
                --text-main: #111827;
                --text-muted: #6B7280;
            }
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: var(--bg); 
                color: var(--text-main);
                margin: 0; min-height: 100vh;
                display: flex; align-items: center; justify-content: center;
            }
            .container {
                max-width: 500px; width: 90%;
                background: var(--card-bg);
                padding: 2.5rem; border-radius: 1rem;
                box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 { font-weight: 600; margin-bottom: 0.5rem; color: var(--text-main); }
            p { color: var(--text-muted); margin-bottom: 2rem; }
            .upload-area {
                border: 2px dashed #D1D5DB;
                border-radius: 0.75rem;
                padding: 2rem;
                cursor: pointer;
                transition: border-color 0.2s, background 0.2s;
                position: relative;
            }
            .upload-area:hover { border-color: var(--primary); background: #F5F3FF; }
            #imagePreview { max-width: 100%; border-radius: 0.5rem; margin-top: 1rem; display: none; }
            #fileInput { display: none; }
            button {
                background: var(--primary); color: white;
                border: none; padding: 0.75rem 1.5rem;
                border-radius: 0.5rem; font-weight: 600;
                cursor: pointer; width: 100%;
                margin-top: 1.5rem; transition: background 0.2s;
            }
            button:hover { background: var(--primary-hover); }
            button:disabled { background: #9CA3AF; cursor: not-allowed; }
            .result {
                margin-top: 1.5rem; padding: 1rem;
                border-radius: 0.5rem; background: #F3F4F6;
                display: none; text-align: left;
            }
            .result h3 { margin-top: 0; font-size: 1.1rem; }
            .prob-bar {
                height: 8px; background: #E5E7EB; border-radius: 4px;
                overflow: hidden; margin-top: 4px;
            }
            .prob-fill { height: 100%; background: var(--primary); }
            .status-badge {
                display: inline-block; padding: 0.25rem 0.75rem;
                border-radius: 9999px; font-size: 0.875rem; font-weight: 500;
                margin-bottom: 1rem;
            }
            .online { background: #DCFCE7; color: #166534; }
            .offline { background: #FEE2E2; color: #991B1B; }
        </style>
    </head>
    <body>
        <div class="container">
            <div style="margin-bottom: 2rem; padding: 1.5rem; background: #EBF5FB; border: 3px solid #1A5276; border-radius: 0.75rem; text-align: left;">
                <h2 style="font-size: 1.2rem; font-weight: 800; margin: 0 0 1rem 0; color: #1A5276; text-align: center; border-bottom: 2px solid #1A5276; padding-bottom: 0.5rem;">TEAM MEMBERS - GROUP 115</h2>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; color: #111827;">
                    <thead>
                        <tr style="text-align: left; border-bottom: 1px solid #1A5276; color: #1A5276;">
                            <th style="padding: 4px;">#</th>
                            <th style="padding: 4px;">Student Name</th>
                            <th style="padding: 4px;">Student ID</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 4px;">1</td>
                            <td style="padding: 4px;"><strong>MAJJIGI JAMBANNA</strong></td>
                            <td style="padding: 4px;">2024AA05721</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px;">2</td>
                            <td style="padding: 4px;"><strong>NIRANJAN KUMAR SHARMA</strong></td>
                            <td style="padding: 4px;">2024AA05405</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px;">3</td>
                            <td style="padding: 4px;"><strong>SRIDEVI THAKKU RAMANAN</strong></td>
                            <td style="padding: 4px;">2024AA05402</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px;">4</td>
                            <td style="padding: 4px;"><strong>TRIPTI</strong></td>
                            <td style="padding: 4px;">2024AA05593</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px;">5</td>
                            <td style="padding: 4px;"><strong>HARSHAL KISHORE PHADAS</strong></td>
                            <td style="padding: 4px;">2024AA05139</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <span class="status-badge """ + ("online" if os.path.exists(MODEL_PATH) else "offline") + """">
                """ + ("Model Online" if os.path.exists(MODEL_PATH) else "Model Offline") + """
            </span>
            <h1>Cat or Dog?</h1>
            <p>Upload an image to see the classification.</p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div id="uploadPrompt">
                    <svg style="width: 48px; height: 48px; color: #9CA3AF; margin-bottom: 1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                    </svg>
                    <div style="font-weight: 500">Click to upload</div>
                    <div style="font-size: 0.875rem; color: #9CA3AF">PNG, JPG, BMP up to 10MB</div>
                </div>
                <img id="imagePreview" src="" alt="Preview">
            </div>
            
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
            <button id="classifyBtn" onclick="classifyImage()" disabled>Classify Image</button>
            
            <div id="result" class="result">
                <h3>Prediction: <span id="predLabel" style="color: var(--primary)">-</span></h3>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem">
                        <span>Cat</span>
                        <span id="catProb">0%</span>
                    </div>
                    <div class="prob-bar"><div id="catFill" class="prob-fill" style="width: 0%"></div></div>
                </div>
                <div style="margin-top: 0.75rem">
                    <div style="display: flex; justify-content: space-between; font-size: 0.875rem">
                        <span>Dog</span>
                        <span id="dogProb">0%</span>
                    </div>
                    <div class="prob-bar"><div id="dogFill" class="prob-fill" style="width: 0%"></div></div>
                </div>
            </div>
        </div>

        <script>
            function previewImage(event) {
                const reader = new FileReader();
                reader.onload = function() {
                    const output = document.getElementById('imagePreview');
                    output.src = reader.result;
                    output.style.display = 'block';
                    document.getElementById('uploadPrompt').style.display = 'none';
                    document.getElementById('classifyBtn').disabled = false;
                }
                reader.readAsDataURL(event.target.files[0]);
            }

            async function classifyImage() {
                const fileInput = document.getElementById('fileInput');
                const btn = document.getElementById('classifyBtn');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) return;

                btn.disabled = true;
                btn.innerText = 'Analyzing...';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    if (response.ok) {
                        resultDiv.style.display = 'block';
                        document.getElementById('predLabel').innerText = data.prediction;
                        
                        const catP = (data.probabilities.cat * 100).toFixed(1);
                        const dogP = (data.probabilities.dog * 100).toFixed(1);
                        
                        document.getElementById('catProb').innerText = catP + '%';
                        document.getElementById('catFill').style.width = catP + '%';
                        
                        document.getElementById('dogProb').innerText = dogP + '%';
                        document.getElementById('dogFill').style.width = dogP + '%';
                    } else {
                        alert('Error: ' + data.detail);
                    }
                } catch (error) {
                    alert('Error connecting to the server.');
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Classify Image';
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health", tags=["System"])
def health_check():
    """
    Standardize health check endpoint for monitoring and orchestration (Docker/K8s).
    Returns model load status, path, and active device.
    """
    return {
        "status": "healthy", 
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "device": str(device)
    }

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint:
    1. Validates model existence
    2. Reads and preprocesses the uploaded image
    3. Performs forward pass through the network
    4. Returns prediction label (Cat/Dog) and confidence scores
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model weights not found. Train the model first.")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        label = "Dog" if predicted.item() == 1 else "Cat"
        
        logger.info(f"Prediction: {label}, Confidence: {confidence.item():.4f}")
        
        return {
            "prediction": label,
            "confidence": float(confidence.item()),
            "probabilities": {
                "cat": float(probabilities[0][0].item()),
                "dog": float(probabilities[0][1].item())
            }
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Instrument with Prometheus
try:
    Instrumentator().instrument(app).expose(app)
except Exception as e:
    logger.warning(f"Could not initialize Prometheus instrumentation: {e}")

if __name__ == "__main__":
    import uvicorn # uvicorn>=0.23.2
    parser = argparse.ArgumentParser(description="Cats vs Dogs API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--model_name", type=str, default="model.pt", help="Model filename (must be in models/ folder)")
    args = parser.parse_args()
    
    # Reload model with the specified name
    load_model(args.model_name)
    
    uvicorn.run(app, host=args.host, port=args.port)

