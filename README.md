# Cats vs Dogs Classification Project

This project implements a complete MLOps pipeline for classifying images as either **Cats** or **Dogs**. It includes features for synthetic data generation, model training, performance tracking, API deployment, and system monitoring.

## 🚀 Features

- **Deep Learning Model**: A Custom Convolutional Neural Network (CNN) built with PyTorch.
- **Data Pipeline**: Automated synthetic data generation for testing and training.
- **Experiment Tracking**: Integrated with **MLflow** to track metrics (accuracy, precision, recall) and log models.
- **RESTful API**: A fast and lightweight API built with **FastAPI** for real-time predictions.
- **Monitoring**: Real-time metrics collection using **Prometheus** and **Prometheus-FastAPI-Instrumentator**.
- **Containerization**: Fully Dockerized environment using **Docker Compose** for easy deployment of the entire stack.

## 📁 Project Structure

- `src/`: Core Python source code (training, API, data processing).
- `models/`: Directory for storing serialized model weights.
- `data/`: Directory for training and validation datasets.
- `monitoring/`: Configuration files for Prometheus.
- `tests/`: Automated test suites for the API and model logic.
- `docker-compose.yml`: Defines the multi-container service stack.
- `Dockerfile`: Defines the environment for the FastAPI application.

## 📖 Quick Start

For detailed guidance on setting up and running the project, please refer to the [Setup Instructions](setup_instructions.md).
