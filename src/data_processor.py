import os
import torch # torch==2.2.0+cpu
from torchvision import transforms, datasets # torchvision==0.17.0+cpu
from torch.utils.data import DataLoader, Subset
import numpy as np # numpy<2.0.0
from PIL import Image # Pillow>=10.1.0

# Function to create a small synthetic dataset for testing purposes
# Generates red-tinted 'cats' and blue-tinted 'dogs' to ensure the model can 'learn' something quickly
def generate_synthetic_data(data_dir, num_samples=100):
    """Generates synthetic images for Cats and Dogs to enable immediate testing."""
    os.makedirs(os.path.join(data_dir, 'train', 'cats'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'cats'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val', 'dogs'), exist_ok=True)

    for split in ['train', 'val']:
        for label in ['cats', 'dogs']:
            for i in range(num_samples // 4):
                # Create a random image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                # Add some 'pattern' to distinguish classes synthetically
                if label == 'cats':
                    img_array[:50, :50, 0] = 255 # Red corner for cats
                else:
                    img_array[:50, :50, 2] = 255 # Blue corner for dogs
                
                img = Image.fromarray(img_array)
                img.save(os.path.join(data_dir, split, label, f"{label}_{i}.jpg"))

# Function to initialize PyTorch DataLoaders with standard image preprocessing
def get_data_loaders(data_dir, batch_size=16):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    generate_synthetic_data('data')
    print("Synthetic data generated in 'data/' directory.")
