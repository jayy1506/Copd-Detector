#!/usr/bin/env python3
"""
Verify that the model issue is fixed
"""

import os
import numpy as np
import cv2
from tensorflow import keras

def load_model():
    """Try to load the retrained model first, then the original"""
    model_paths = ['best_copd_model_retrained.h5', 'best_copd_model.h5']
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Loading model: {path}")
            try:
                model = keras.models.load_model(path)
                print(f"Model loaded successfully from {path}!")
                return model, path
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    print("No model files found!")
    return None, None

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

def test_model_predictions(model, test_dir, num_samples=10):
    """Test model predictions on sample images"""
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    files = os.listdir(test_dir)
    # Filter for image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {test_dir}")
        return
    
    print(f"Found {len(image_files)} images in {test_dir}")
    
    # Take a sample
    if len(image_files) > num_samples:
        import random
        image_files = random.sample(image_files, num_samples)
    
    print(f"\nTesting {len(image_files)} sample images...")
    print("-" * 80)
    print(f"{'Filename':35} | {'Expected':8} | {'Predicted':9} | {'Score':8} | {'Status'}")
    print("-" * 80)
    
    correct_predictions = 0
    total_predictions = 0
    
    for filename in image_files:
        filepath = os.path.join(test_dir, filename)
        
        # Determine expected label based on filename
        if filename.startswith('Normal') or filename.startswith('Normal ('):
            expected_label = "Normal"
        elif filename.startswith('COVID') or filename.startswith('COVID (') or filename.startswith('Emphysema'):
            expected_label = "COPD"
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
            if expected_label != "Unknown":
                total_predictions += 1
                if predicted_label == expected_label:
                    correct_predictions += 1
                    status = "✓ CORRECT"
                else:
                    status = "✗ WRONG"
            else:
                status = "? UNKNOWN"
            
            print(f"{filename:35} | {expected_label:8} | {predicted_label:9} | {probability:7.2f}% | {status}")
            
        except Exception as e:
            print(f"{filename:35} | Error: {str(e)}")
    
    print("-" * 80)
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2f}%)")
    
    return correct_predictions, total_predictions

def main():
    print("COPD Model Verification Tool")
    print("=" * 50)
    
    # Load model
    model, model_path = load_model()
    if model is None:
        return
    
    print(f"Using model: {model_path}")
    print(f"Model Input Shape: {model.input_shape}")
    print(f"Model Output Shape: {model.output_shape}")
    
    # Test predictions
    print("\nModel Prediction Verification:")
    print("=" * 50)
    
    test_dir = os.path.join('.', 'test')
    if os.path.exists(test_dir):
        correct, total = test_model_predictions(model, test_dir, num_samples=20)
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"\nOverall Accuracy: {accuracy:.2f}%")
        else:
            print("\nNo valid predictions to evaluate")

if __name__ == "__main__":
    main()