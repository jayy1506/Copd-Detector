#!/usr/bin/env python3
"""
Quick script to evaluate the trained COPD detection model with minimal data
"""

print("Starting quick model evaluation...")

import os
import numpy as np
import cv2
from tensorflow import keras

# Constants
IMG_SIZE = 224

def load_sample_test_data(data_dir):
    """Load a small sample of test data from directory"""
    print("Loading sample test data...")
    test_dir = os.path.join(data_dir, 'test')
    
    images = []
    labels = []
    
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        print(f"Found {len(files)} files in test directory")
        
        # Limit to first 5 files for very quick testing
        files = files[:5]
        print(f"Processing only {len(files)} files for quick test")
        
        for filename in files:
            img_path = os.path.join(test_dir, filename)
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
                        
                    print(f"Loaded: {filename} -> Label: {labels[-1]}")
                else:
                    print(f"Failed to load: {filename}")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                
        print(f"Loaded {len(images)} test images")
    
    return np.array(images), np.array(labels)

def quick_evaluate():
    """Quick evaluation function"""
    print("="*50)
    print("Quick COPD Model Evaluation")
    print("="*50)
    
    # Check if model exists
    model_path = 'best_copd_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    print("Loading trained model...")
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load sample test data
    data_dir = r"c:\Users\jthak\OneDrive\Desktop\chest-xray"
    X_test, y_test = load_sample_test_data(data_dir)
    
    if len(X_test) == 0:
        print("No test data loaded. Exiting.")
        return
    
    # Normalize pixel values
    X_test = X_test.astype('float32') / 255.0
    
    # Make predictions
    print("Making predictions...")
    try:
        predictions = model.predict(X_test)
        print("Predictions completed!")
        
        # Display results
        print("\nResults:")
        for i in range(len(X_test)):
            pred_prob = predictions[i][0]
            pred_label = "COPD" if pred_prob > 0.5 else "Normal"
            true_label = "COPD" if y_test[i] == 1 else "Normal"
            print(f"Sample {i+1}: True={true_label}, Predicted={pred_label} (Prob: {pred_prob:.4f})")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

    print("\nQuick evaluation completed!")

if __name__ == "__main__":
    quick_evaluate()