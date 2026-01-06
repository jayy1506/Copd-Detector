import os
import numpy as np
import cv2

IMG_SIZE = 224

def load_data_from_directory(directory_path):
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
        count = 0
        for filename in files[:10]:  # Only process first 10 files for testing
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
                    # Handle files that start with "Normal (" or "COVID ("
                    elif filename.startswith('Normal ('):
                        images.append(img)
                        labels.append(0)  # Normal class
                    elif filename.startswith('COVID ('):
                        images.append(img)
                        labels.append(1)  # COPD class
                    
                    count += 1
                    print(f"Processed image: {filename}")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        print(f"Finished processing {count} images from {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")
    
    return images, labels

def main():
    print("Simple test script started")
    data_dir = r"c:\Users\jthak\OneDrive\Desktop\chest-xray"
    
    # Test loading training data
    train_images, train_labels = load_data_from_directory(os.path.join(data_dir, 'train'))
    print(f"Loaded {len(train_images)} training images")
    
    print("Simple test script completed")

if __name__ == "__main__":
    main()