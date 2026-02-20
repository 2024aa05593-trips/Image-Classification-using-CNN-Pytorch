import sys
import os
# Add project root to path to enable imports from the 'src' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch # torch==2.2.0+cpu
import numpy as np # numpy<2.0.0
import os
import pytest # pytest>=7.4.2
from src.data_processor import generate_synthetic_data, get_data_loaders
from src.train import SimpleCNN

# Test the synthetic data generation logic
def test_data_generation():
    data_dir = 'test_data'
    # Generate a small sample size for quick unit testing
    generate_synthetic_data(data_dir, num_samples=8)
    assert os.path.exists(os.path.join(data_dir, 'train', 'cats'))
    assert os.path.exists(os.path.join(data_dir, 'train', 'dogs'))
    
    # Cleanup temporary test directory
    import shutil
    shutil.rmtree(data_dir)

# Test that DataLoaders correctly load and batch images
def test_data_loaders():
    data_dir = 'test_data'
    generate_synthetic_data(data_dir, num_samples=16)
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=4)
    
    # Verify the shape of the batches (BatchSize, Channels, Height, Width)
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    
    import shutil
    shutil.rmtree(data_dir)

# Test the model's forward pass logic in isolation
def test_model_forward():
    model = SimpleCNN()
    # Create a dummy input tensor matching the expected input shape
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    # Model should output scores for 2 classes (Cat, Dog)
    assert output.shape == (1, 2)

# Test the high-level prediction and confidence score calculation
def test_prediction_logic():
    model = SimpleCNN()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Verify outputs are valid class indices and probability ranges
    assert predicted.item() in [0, 1]
    assert 0 <= confidence.item() <= 1
