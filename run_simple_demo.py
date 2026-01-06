#!/usr/bin/env python3
"""
Simple demo script for COPD detection model
"""

print("="*60)
print("COPD DETECTION MODEL - SIMPLE DEMO")
print("="*60)

import os
import numpy as np
from tensorflow import keras

# Check if model exists
model_path = 'best_copd_model.h5'
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Trained model found: {model_path} ({size_mb:.1f} MB)")
    
    # Load model
    print("Loading model...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    
    # Show model info
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Show sample prediction (just as demonstration)
    print("\nGenerating sample prediction...")
    # Create a dummy input (in practice, you would load and preprocess a real X-ray image)
    dummy_input = np.random.rand(1, 224, 224, 3)
    prediction = model.predict(dummy_input)
    print(f"Sample prediction: {prediction[0][0]:.4f}")
    print("(Values close to 0 = Normal, Values close to 1 = COPD)")
    
else:
    print("Trained model not found. Please run the full training script first.")

print("\n" + "="*60)
print("SIMPLE DEMO COMPLETED")
print("="*60)