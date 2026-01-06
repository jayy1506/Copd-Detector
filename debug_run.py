#!/usr/bin/env python3
"""
Debug script to run the COPD detection model step by step
"""

print("Debug script started")

# Import required modules
import os
import numpy as np
import cv2

# Constants
IMG_SIZE = 224

def load_data_from_directory(directory_path, max_samples=None):
    """
    Load and preprocess images from a directory
    """
    print(f"Loading data from directory: {directory_path}")
    images = []
    labels = []
    
    if os.path.exists(directory_path):
        print(f"Directory exists. Loading files...")
        files = os.listdir(directory_path)
        print(f"Found {len(files)} files in directory")
        
        # Limit files if max_samples is specified
        if max_samples is not None:
            files = files[:max_samples]
            print(f"Limiting to {len(files)} files for testing")
        
        count = 0
        for filename in files:
            img_path = os.path.join(directory_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Assign labels based on filename prefixes
                    if filename.startswith('Normal'):
                        images.append(img)
                        labels.append(0)  # Normal class
                    elif filename.startswith('COVID') or filename.startswith('Emphysema'):
                        images.append(img)
                        labels.append(1)  # COPD class
                    
                    count += 1
                    if count % 10 == 0:  # Changed from 100 to 10 for faster feedback
                        print(f"Processed {count} images...")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        print(f"Finished processing {count} images from {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")
    
    return images, labels

def debug_main():
    """
    Debug main function
    """
    print("="*50)
    print("DEBUG: COPD Detection using Chest X-Ray Images")
    print("="*50)
    print("Starting debug main function...")
    
    # Load and preprocess data
    data_dir = r"c:\Users\jthak\OneDrive\Desktop\chest-xray"
    print(f"Data directory: {data_dir}")
    
    # Test with very small sample size
    max_samples = 10
    print(f"Loading training data with max_samples={max_samples}...")
    train_images, train_labels = load_data_from_directory(os.path.join(data_dir, 'train'), max_samples)
    
    print(f"Loaded {len(train_images)} training images")
    if len(train_images) > 0:
        print(f"Training - Normal images: {np.sum(np.array(train_labels) == 0)}")
        print(f"Training - COPD images: {np.sum(np.array(train_labels) == 1)}")
    
    print("Debug script completed")

if __name__ == "__main__":
    debug_main()