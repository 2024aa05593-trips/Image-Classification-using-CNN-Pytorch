import os
import random
import shutil

# Utility function to split a single dataset into training and validation sets
# Useful when only raw 'train' data is provided and a separate 'val' folder is needed
def split_train_val(data_dir, val_split=0.2):
    """
    Moves a random percentage of images from train folders to val folders.
    This prevents duplicates and ensures a clean split.
    """
    for label in ['cats', 'dogs']:
        train_path = os.path.join(data_dir, 'train', label)
        val_path = os.path.join(data_dir, 'val', label)
        
        os.makedirs(val_path, exist_ok=True)
        
        # Get all images in train folder
        images = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
        
        # Calculate number of images to move
        num_val = int(len(images) * val_split)
        
        # Randomly select images for validation
        val_images = random.sample(images, num_val)
        
        print(f"Moving {num_val} images from train/{label} to val/{label}...")
        
        for img_name in val_images:
            src_file = os.path.join(train_path, img_name)
            dst_file = os.path.join(val_path, img_name)
            
            # Move the file (replaces if exists in destination)
            shutil.move(src_file, dst_file)

if __name__ == "__main__":
    # You can change the split percentage here (0.2 = 20%)
    split_train_val('data', val_split=0.2)
    print("Dataset split complete.")
