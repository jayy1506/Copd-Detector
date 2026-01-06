#!/usr/bin/env python3
"""
Simple test script to verify Grad-CAM functionality
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

def test_gradcam_basic():
    """Test basic Grad-CAM functionality"""
    print("="*50)
    print("Grad-CAM Basic Test")
    print("="*50)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Initialize model with dummy input
    print("Initializing model...")
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    try:
        # First do a forward pass to build the computational graph
        _ = model(dummy_input)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Check model structure
    print("\nModel structure:")
    print(f"Number of layers: {len(model.layers)}")
    
    # Check first layer (should be ResNet)
    resnet_model = model.layers[0]
    print(f"First layer type: {type(resnet_model).__name__}")
    print(f"ResNet layers: {len(resnet_model.layers)}")
    
    # Look for convolutional layers
    print("\nLooking for convolutional layers...")
    conv_layers = []
    for i, layer in enumerate(resnet_model.layers):
        if 'conv' in layer.name.lower() and layer.name.endswith('conv'):
            conv_layers.append((i, layer.name))
    
    print(f"Found {len(conv_layers)} convolutional layers")
    if conv_layers:
        print("Last 10 conv layers:")
        for i, (idx, name) in enumerate(conv_layers[-10:]):
            print(f"  {name}")
        
        # Try to use a middle layer
        test_layer = conv_layers[len(conv_layers)//2][1]  # Middle layer
        print(f"\nTesting with layer: {test_layer}")
        
        try:
            # Test if we can access the layer
            layer_obj = resnet_model.get_layer(test_layer)
            print(f"Successfully accessed layer: {test_layer}")
            
            # Try to build a simple gradient model
            print("Building gradient model...")
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[layer_obj.output, model.output]
            )
            print("Gradient model built successfully!")
            
            # Test with a sample input
            print("Testing gradient computation...")
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(test_input)
                loss = predictions[0]
            
            grads = tape.gradient(loss, conv_outputs)
            print("Gradient computation successful!")
            
            print("\n✅ Grad-CAM basic functionality test PASSED")
            
        except Exception as e:
            print(f"❌ Error with layer {test_layer}: {e}")
            print("\n❌ Grad-CAM basic functionality test FAILED")
    else:
        print("❌ No convolutional layers found")
        print("\n❌ Grad-CAM basic functionality test FAILED")

if __name__ == "__main__":
    test_gradcam_basic()