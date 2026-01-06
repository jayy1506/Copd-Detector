import os
import numpy as np
import cv2

IMG_SIZE = 224

def load_data_from_directory(directory_path, max_samples=50):
    """
    Load and preprocess images from a directory
    """
    print(f"Loading data from directory: {directory_path}")
    images = []
    labels = []
    normal_count = 0
    copd_count = 0
    
    if os.path.exists(directory_path):
        print(f"Directory exists. Loading files...")
        files = os.listdir(directory_path)
        print(f"Found {len(files)} files in directory")
        
        # Limit files if max_samples is specified
        if max_samples is not None:
            # Randomly sample files to get a better distribution
            import random
            if len(files) > max_samples:
                files = random.sample(files, max_samples)
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
                        normal_count += 1
                    elif filename.startswith('COVID') or filename.startswith('Emphysema'):
                        images.append(img)
                        labels.append(1)  # COPD class
                        copd_count += 1
                    # Handle files that start with "Normal (" or "COVID ("
                    elif filename.startswith('Normal ('):
                        images.append(img)
                        labels.append(0)  # Normal class
                        normal_count += 1
                    elif filename.startswith('COVID ('):
                        images.append(img)
                        labels.append(1)  # COPD class
                        copd_count += 1
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} images...")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        print(f"Finished processing {count} images from {directory_path}")
        print(f"Normal images: {normal_count}, COPD images: {copd_count}")
    else:
        print(f"Directory does not exist: {directory_path}")
    
    return images, labels

def main():
    print("Checking data loading...")
    data_dir = r"c:\Users\jthak\OneDrive\Desktop\chest-xray"
    
    # Test loading training data
    train_images, train_labels = load_data_from_directory(os.path.join(data_dir, 'train'), max_samples=100)
    print(f"Loaded {len(train_images)} training images")
    print(f"Training labels distribution: {np.bincount(train_labels)}")
    
    # Test loading test data
    test_images, test_labels = load_data_from_directory(os.path.join(data_dir, 'test'), max_samples=50)
    print(f"Loaded {len(test_images)} test images")
    print(f"Test labels distribution: {np.bincount(test_labels) if len(test_labels) > 0 else 'No labels'}")

if __name__ == "__main__":
    main()