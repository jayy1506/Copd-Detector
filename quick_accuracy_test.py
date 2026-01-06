#!/usr/bin/env python3
"""
Quick script to evaluate the COPD detection model accuracy
"""

import os
import numpy as np
import cv2
from tensorflow import keras

# Constants
IMG_SIZE = 224

def load_sample_test_data(data_dir, max_samples=20):
    """Load a small sample of test data from directory for quick evaluation"""
    print("Loading sample test data...")
    test_dir = os.path.join(data_dir, 'test')
    
    images = []
    labels = []
    
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        print(f"Found {len(files)} files in test directory")
        
        # Limit to max_samples files for quicker testing
        files = files[:max_samples]
        print(f"Processing {len(files)} files for quick test")
        
        processed_count = 0
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
                    
                    processed_count += 1
                    if processed_count % 5 == 0:
                        print(f"Processed {processed_count}/{len(files)} images...")
                        
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                
        print(f"Successfully loaded {len(images)} test images")
        print(f"Normal images: {np.sum(np.array(labels) == 0)}")
        print(f"COPD images: {np.sum(np.array(labels) == 1)}")
    
    return np.array(images), np.array(labels)

def quick_accuracy_test():
    """Quick accuracy test function"""
    print("="*50)
    print("Quick COPD Model Accuracy Test")
    print("="*50)
    
    # Check if model exists
    model_paths = [
        'best_copd_model_retrained.h5',  # Try retrained model first
        'best_copd_model.h5',
        'final_copd_model.h5'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("No model files found!")
        return
    
    print(f"Loading trained model: {model_path}...")
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load sample test data
    data_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory
    X_test, y_test = load_sample_test_data(data_dir, max_samples=20)  # Use only 20 samples for quick test
    
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
        
        # Calculate accuracy
        predicted_classes = (predictions.flatten() > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == y_test)
        
        print(f"\nAccuracy Results:")
        print(f"Total samples: {len(X_test)}")
        print(f"Correct predictions: {np.sum(predicted_classes == y_test)}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show detailed results
        print(f"\nDetailed Results:")
        for i in range(len(X_test)):
            pred_prob = predictions[i][0]
            pred_label = "COPD" if pred_prob > 0.5 else "Normal"
            true_label = "COPD" if y_test[i] == 1 else "Normal"
            correct = "✓" if pred_label == true_label else "✗"
            print(f"Sample {i+1:2d}: True={true_label:>5s}, Pred={pred_label:>5s} (Prob: {pred_prob:.4f}) {correct}")
        
        # Calculate class-specific accuracy
        copd_mask = y_test == 1
        normal_mask = y_test == 0
        
        if np.sum(copd_mask) > 0:
            copd_accuracy = np.mean(predicted_classes[copd_mask] == y_test[copd_mask])
            print(f"\nCOPD Detection Accuracy: {copd_accuracy:.4f} ({copd_accuracy*100:.2f}%) "
                  f"({np.sum(predicted_classes[copd_mask] == y_test[copd_mask])}/{np.sum(copd_mask)} correct)")
        
        if np.sum(normal_mask) > 0:
            normal_accuracy = np.mean(predicted_classes[normal_mask] == y_test[normal_mask])
            print(f"Normal Detection Accuracy: {normal_accuracy:.4f} ({normal_accuracy*100:.2f}%) "
                  f"({np.sum(predicted_classes[normal_mask] == y_test[normal_mask])}/{np.sum(normal_mask)} correct)")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

    print("\nQuick accuracy test completed!")

if __name__ == "__main__":
    quick_accuracy_test()