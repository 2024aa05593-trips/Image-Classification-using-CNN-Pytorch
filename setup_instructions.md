# Step-by-Step Running Instructions

This guide provides detailed instructions for setting up, training, and validating the Cats vs Dogs Classification project in two environments: **Local** and **Docker**.

---

## 🛠️ Section 1: Running Locally

Follow these steps to run the training and API services directly on your host machine.

### 1. Environment Setup
Prepare your Python environment and install dependencies:
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Model Training & Data Generation
Generate synthetic data and train the initial model. By default, the script logs experiments to the MLflow server at `http://localhost:5000`.

```bash
# Basic training (defaults to: epochs=5, lr=0.001, experiment="Cats_vs_Dogs_Classification")
python src/train.py

# Custom training session
I. python src/train.py --epochs 2 --experiment "Local_Experiment_v1" --model_name "local_model.pt"

# You can also specify the tracking URI if your MLflow server is on a different port/address
python src/train.py --tracking_uri "http://localhost:5000" --experiment "Remote_Tracking_Test"
```
> [!NOTE]
> If the MLflow server is not running, the script will fall back to local logging in the `mlruns/` directory.

*   **Validation**: Check the `models/` directory for your saved `.pt` file and the MLflow UI (if running) for experiment results.

### 3. Local Model Testing
Verify the model logic and data processing before starting the server:
```bash
# Run the full test suite
II. pytest tests/ -v
```

### 4. API Execution
Start the FastAPI server locally to access the prediction service and web interface:
```bash
# Start the application (defaults to: model_name="model.pt")
III. python src/app.py

# Start the application with a specific model (e.g. your latest Feb 20 model)
python src/app.py --model_name "local_modelFeb20.pt"
```
*  1.  **Interactive Web Interface**: Open [http://localhost:8000/](http://localhost:8000/) in your browser.
*  2. **API Documentation (Swagger)**: Access [http://localhost:8000/docs](http://localhost:8000/docs) to test endpoints manually.
*  3. **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
*  4. Run command mlflow ui   before accessing http://127.0.0.1:5000    
*  5. http://127.0.0.1:5000
*  6. Prometheus raw metrics http://localhost:8000/metrics   or  curl http://localhost:8000/health

### 5. Client-Side Automation
Test the API programmatically using the included client script:
```bash
# Ensure the server is running in another terminal
IV. python tests/test_api_client.py
```

---

## 🐳 Section 2: Running in Docker

### 0. Verify Docker Status
Before starting, ensure Docker Desktop is running. In your terminal, run:
```bash
# This should show the Docker version and system info without errors
docker info
```
> [!IMPORTANT]
> If you get a "failed to connect to the docker API" error, start **Docker Desktop** from your Start menu and wait for the status icon to turn green.

### 1. Build the Docker Image
```bash
docker build -t cats-dogs-classifier:latest .
```

### 2. Individual Container Validation (Optional)
Verify the image runs correctly in isolation:
```bash
docker run -d -p 8000:8000 --name classifier cats-dogs-classifier:latest
docker logs classifier
curl http://localhost:8000/health
docker stop classifier && docker rm classifier
```

### 3. Full Stack with Docker Compose (Recommended)
Launch the entire environment including the API, MLflow tracking server, and Prometheus monitoring:
```bash
# Start all services
docker-compose up --build -d
```

IMP : 
set  $env:MLFLOW_TRACKING_URI="http://localhost:5000"


Notes : 
docker-compose up -d
    Only starts the containers. If you already have an old version of the image on your computer, it will use that old version even if you changed your Python code.
    
docker-compose up --build -d
 This tells Docker to check for code changes and rebuild the image before starting the containers.



### 4. Service Dashboards
Access the following URLs to verify the system status:

| Service | URL | Validation Point |
| :--- | :--- | :--- |
| **Web Interface** | [http://localhost:8000](http://localhost:8000) | Drag & Drop UI for classification |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | View training metrics and model versions |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Monitor system performance and usage |

Note Prometheus : http://localhost:9090/targets

### 5. Cleanup
To stop and remove all services:
```bash
docker-compose down
```

---

## 🔍 Troubleshooting

### 1. Port 8000 Already in Use
If you see `[Errno 10048]` when starting `app.py`, it means the API is already running in Docker or another process.
*   **Check**: Run `netstat -ano | findstr :8000` (CMD) or `Get-NetTCPConnection -LocalPort 8000` (PowerShell) to see if it's active.
*   **Fix (Docker)**: Run `docker-compose down` to stop all containers.
*   **Fix (Manual)**: Run this in PowerShell to kill the process forcefully:
    ```powershell
    Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
    ```

### 2. MLflow Server Not Reachable
If `train.py` reports the server is not reachable:
*   Ensure the MLflow container is running: `docker ps`
*   Verify you can open [http://localhost:5000](http://localhost:5000) in your browser.
*   Check if your firewall is blocking port 5000.
