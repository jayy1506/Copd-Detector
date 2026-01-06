#!/usr/bin/env python3
"""
Diagnostic script to test Grad-CAM functionality
"""

import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf

def load_model():
    """Load the trained COPD detection model"""
    model_paths = [
        'best_copd_model_retrained.h5',
        'best_copd_model.h5',
        'final_copd_model.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}...")
                model = keras.models.load_model(model_path)
                print(f"Model loaded successfully!")
                return model
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                continue
    
    print("No model files found!")
    return None

def find_last_conv_layer(model):
    """Find the last convolutional layer in the ResNet model"""
    try:
        # Print model structure to understand its architecture
        print("Model layers:")
        for i, layer in enumerate(model.layers):
            print(f"  {i}: {layer.name} - {type(layer).__name__}")
        
        resnet_model = model.layers[0]  # First layer is the ResNet50 model
        print(f"ResNet model type: {type(resnet_model).__name__}")
        print(f"ResNet model layers count: {len(resnet_model.layers)}")
        
        # Print ResNet layers
        print("ResNet layers:")
        for i, layer in enumerate(resnet_model.layers):
            print(f"  {i}: {layer.name} - {type(layer).__name__} - Output shape: {getattr(layer, 'output_shape', 'N/A')}")
        
        # Look for convolutional layers from the end
        for layer in reversed(resnet_model.layers):
            print(f"Checking layer: {layer.name} - Type: {type(layer).__name__}")
            # Look for the specific layer we want to use for Grad-CAM
            if 'conv' in layer.name.lower() and hasattr(layer, 'output_shape') and len(getattr(layer, 'output_shape', [])) == 4:
                print(f"Found suitable conv layer: {layer.name}")
                return layer.name
        
        print("No suitable convolutional layer found")
        return None
    except Exception as e:
        print(f"Error finding last conv layer: {e}")
        return None

def main():
    print("="*50)
    print("Grad-CAM Diagnostic Tool")
    print("="*50)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Find last convolutional layer
    print("\nFinding last convolutional layer...")
    last_conv_layer = find_last_conv_layer(model)
    
    if last_conv_layer:
        print(f"\nSUCCESS: Found last conv layer: {last_conv_layer}")
    else:
        print("\nFAILURE: Could not find suitable convolutional layer")

if __name__ == "__main__":
    main()