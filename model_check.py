#!/usr/bin/env python3
"""
Simple script to check if we can load the trained model
"""

print("Checking model loading capability...")

import os
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")

# Check if model exists
model_path = 'best_copd_model.h5'
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model file found: {model_path} ({size_mb:.1f} MB)")
    
    try:
        print("Attempting to load model...")
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print("Model summary:")
        model.summary()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
else:
    print(f"✗ Model file not found: {model_path}")

print("Model check completed.")