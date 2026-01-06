#!/usr/bin/env python3
"""
Diagnostic tool to analyze COPD detection model behavior
"""

import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf

def load_model():
    """Load the trained model"""
    model_path = 'best_copd_model.h5'
    if os.path.exists(model_path):
        print("Loading model...")
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    else:
        print(f"Model file {model_path} not found!")
        return None

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
        
    # Resize to model input size
    img = cv2.resize(img, (224, 224))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def analyze_predictions(model, image_dir, num_samples=10):
    """Analyze model predictions on a sample of images"""
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return
    
    files = os.listdir(image_dir)
    # Filter for image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Take a sample
    if len(image_files) > num_samples:
        import random
        image_files = random.sample(image_files, num_samples)
    
    print(f"\nAnalyzing {len(image_files)} sample images...")
    print("-" * 60)
    
    normal_count = 0
    copd_count = 0
    
    for filename in image_files:
        filepath = os.path.join(image_dir, filename)
        
        # Determine if this should be normal or COPD based on filename
        if filename.startswith('Normal') or filename.startswith('Normal ('):
            expected_label = "Normal"
            normal_count += 1
        elif filename.startswith('COVID') or filename.startswith('COVID (') or filename.startswith('Emphysema'):
            expected_label = "COPD"
            copd_count += 1
        else:
            expected_label = "Unknown"
        
        # Preprocess image
        processed_img = preprocess_image(filepath)
        if processed_img is None:
            continue
            
        # Make prediction
        try:
            prediction = model.predict(processed_img)[0][0]
            probability = prediction * 100
            
            if prediction > 0.5:
                predicted_label = "COPD"
            else:
                predicted_label = "Normal"
            
            # Check if prediction matches expectation
            match = "✓" if predicted_label == expected_label else "✗"
            
            print(f"{match} {filename:30} | Expected: {expected_label:6} | Predicted: {predicted_label:6} | Score: {probability:6.2f}%")
            
        except Exception as e:
            print(f"✗ {filename:30} | Error: {str(e)}")
    
    print("-" * 60)
    print(f"Expected distribution: Normal={normal_count}, COPD={copd_count}")
    
    return normal_count, copd_count

def check_data_distribution():
    """Check the actual distribution of files in the dataset"""
    dirs = ['train', 'val', 'test']
    
    print("\nDataset Distribution Analysis:")
    print("=" * 50)
    
    for directory in dirs:
        dir_path = os.path.join('.', directory)
        if not os.path.exists(dir_path):
            print(f"{directory:8}: Directory not found")
            continue
            
        files = os.listdir(dir_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        normal_count = 0
        copd_count = 0
        
        for filename in image_files:
            if filename.startswith('Normal') or filename.startswith('Normal ('):
                normal_count += 1
            elif filename.startswith('COVID') or filename.startswith('COVID (') or filename.startswith('Emphysema'):
                copd_count += 1
        
        total = len(image_files)
        print(f"{directory:8}: Total={total:4} | Normal={normal_count:4} | COPD={copd_count:4}")

def main():
    print("COPD Model Diagnostic Tool")
    print("=" * 50)
    
    # Check data distribution
    check_data_distribution()
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    print(f"\nModel Input Shape: {model.input_shape}")
    print(f"Model Output Shape: {model.output_shape}")
    
    # Analyze predictions on test set
    print("\nModel Prediction Analysis:")
    print("=" * 50)
    
    # Test on a sample from each directory
    for directory in ['test']:
        dir_path = os.path.join('.', directory)
        if os.path.exists(dir_path):
            print(f"\nAnalyzing {directory} directory:")
            analyze_predictions(model, dir_path, num_samples=20)

if __name__ == "__main__":
    main()