#!/usr/bin/env python3
"""
Script to evaluate the trained COPD detection model
"""

print("Starting model evaluation...")

import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32

def load_test_data(data_dir):
    """Load test data from directory"""
    print("Loading test data...")
    test_dir = os.path.join(data_dir, 'test')
    
    images = []
    labels = []
    
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        print(f"Found {len(files)} files in test directory")
        
        # Limit to first 50 files for quicker testing
        files = files[:50]
        
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
                        
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                
        print(f"Loaded {len(images)} test images")
        print(f"Normal images: {np.sum(np.array(labels) == 0)}")
        print(f"COPD images: {np.sum(np.array(labels) == 1)}")
    
    return np.array(images), np.array(labels)

def create_test_generator(X_test, y_test):
    """Create test data generator"""
    # Normalize pixel values
    X_test = X_test.astype('float32') / 255.0
    
    # Create data generator
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_generator

def evaluate_model(model, test_generator, y_test):
    """Evaluate the model"""
    print("Evaluating model...")
    
    # Predictions
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'COPD']))
    
    return y_pred, y_pred_prob

def main():
    """Main evaluation function"""
    print("="*50)
    print("COPD Model Evaluation")
    print("="*50)
    
    # Check if model exists
    model_path = 'best_copd_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    print("Loading trained model...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Load test data
    data_dir = r"c:\Users\jthak\OneDrive\Desktop\chest-xray"
    X_test, y_test = load_test_data(data_dir)
    
    if len(X_test) == 0:
        print("No test data loaded. Exiting.")
        return
    
    # Create test generator
    test_gen = create_test_generator(X_test, y_test)
    
    # Evaluate model
    y_pred, y_pred_prob = evaluate_model(model, test_gen, y_test)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()