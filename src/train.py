import torch # torch==2.2.0+cpu
import torch.nn as nn
import torch.optim as optim
import mlflow # mlflow>=2.10.0
import mlflow.pytorch
import requests
import os
import sys
import argparse
import warnings

# Suppress MLflow warning about artifact_path deprecation
warnings.filterwarnings("ignore", message=".*artifact_path is deprecated.*")


Num_epochs = 1

# Add project root to path to fix ModuleNotFoundError: No module named 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import get_data_loaders, generate_synthetic_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay # scikit-learn>=1.4.0
import numpy as np # numpy<2.0.0
import matplotlib.pyplot as plt # matplotlib>=3.8.0

# Simple CNN Architecture definition for training
# This matches the structure defined in the inference API (src/app.py)
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

def train_model(epochs=Num_epochs, lr=0.001, experiment_name="Cats_vs_Dogs_Classification", model_name="model.pt", tracking_uri="http://localhost:5000"):
    """
    Core training function:
    1. Sets up MLflow tracking (server or local fallback)
    2. Initializes model, loss function, and optimizer
    3. Handles synthetic data generation if datasets are missing
    4. Executes training loop and logs metrics/artifacts to MLflow
    """
    # Set tracking URI to point to the MLflow server (Docker)
    if tracking_uri:
        try:
            # Check if MLflow server is reachable via HTTP request
            requests.get(tracking_uri, timeout=2)
            mlflow.set_tracking_uri(tracking_uri)
            print(f"Successfully connected to MLflow server at: {mlflow.get_tracking_uri()}")
        except Exception:
            # Fallback to local SQLite tracking if server is unavailable
            local_uri = "sqlite:///mlflow.db"
            mlflow.set_tracking_uri(local_uri)
            print(f"Warning: MLflow server at {tracking_uri} is not reachable.")
            print(f"Falling back to local logging at: {mlflow.get_tracking_uri()}")

    # Setup device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists('data/train'):
        generate_synthetic_data('data')
    
    train_loader, val_loader = get_data_loaders('data')

    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", 16)
        
        losses = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
            mlflow.log_metric("train_loss", running_loss/len(train_loader), step=epoch)
            losses.append(running_loss/len(train_loader))

        # Generate Loss Curve Plot for Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), losses, marker='o')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        os.makedirs('artifacts', exist_ok=True)
        plt.savefig('artifacts/loss_curve.png')
        plt.close()
        mlflow.log_artifact('artifacts/loss_curve.png')

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f"Validation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Generate Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
        plt.figure(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig('artifacts/confusion_matrix.png')
        plt.close()
        mlflow.log_artifact('artifacts/confusion_matrix.png')

        # Model Saving Logic:
        # 1. Save weights locally as .pt file
        # 2. Package model via MLflow with environment requirements
        # 3. Log the .pt file as an explicit artifact for easy manual download
        os.makedirs('models', exist_ok=True)
        save_path = os.path.join('models', model_name)
        torch.save(model.state_dict(), save_path)
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            artifact_path="model",
            pip_requirements=[
                "torch==2.2.0",
                "torchvision==0.17.0",
                "mlflow"
            ]
        )
        # Explicitly log the .pt file as an artifact for consistency in the UI
        mlflow.log_artifact(save_path, artifact_path="weights")
        print(f"Model saved to models/{model_name} and logged to MLflow weights/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs CNN")
    parser.add_argument("--epochs", type=int, default=Num_epochs, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--experiment", type=str, default="Cats_vs_Dogs_Classification", help="MLflow experiment name")
    parser.add_argument("--model_name", type=str, default="model.pt", help="Filename to save the model as (e.g. model_v1.pt)")
    parser.add_argument("--tracking_uri", type=str, default="http://localhost:5000", help="MLflow tracking URI")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, lr=args.lr, experiment_name=args.experiment, model_name=args.model_name, tracking_uri=args.tracking_uri)
