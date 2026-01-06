#!/usr/bin/env python3
"""
Detailed model diagnostics to understand structure for Grad-CAM
"""

import os
import numpy as np
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

def detailed_model_analysis():
    """Analyze model structure in detail"""
    print("="*60)
    print("DETAILED MODEL ANALYSIS FOR GRAD-CAM")
    print("="*60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    print(f"\nMain model type: {type(model).__name__}")
    print(f"Main model layers count: {len(model.layers)}")
    
    # Analyze each layer
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i}: {layer.name} ({type(layer).__name__})")
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            print(f"  Sub-layers: {len(layer.layers)}")
            # Show first few and last few sub-layers
            sub_layers = layer.layers
            if len(sub_layers) <= 10:
                for j, sub_layer in enumerate(sub_layers):
                    print(f"    {j}: {sub_layer.name} ({type(sub_layer).__name__})")
            else:
                print("    First 5 sub-layers:")
                for j in range(5):
                    sub_layer = sub_layers[j]
                    print(f"      {j}: {sub_layer.name} ({type(sub_layer).__name__})")
                print("    ...")
                print("    Last 5 sub-layers:")
                for j in range(len(sub_layers)-5, len(sub_layers)):
                    sub_layer = sub_layers[j]
                    print(f"      {j}: {sub_layer.name} ({type(sub_layer).__name__})")
    
    # Initialize model
    print("\nInitializing model...")
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    try:
        _ = model(dummy_input)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Try to find ResNet model
    print("\nIdentifying ResNet component...")
    resnet_model = None
    resnet_index = -1
    for i, layer in enumerate(model.layers):
        if 'resnet' in layer.name.lower() or type(layer).__name__ == 'Functional':
            resnet_model = layer
            resnet_index = i
            print(f"Found ResNet-like model at index {i}: {layer.name}")
            break
    
    if resnet_model is None:
        print("Could not identify ResNet component")
        return
    
    # Analyze ResNet model specifically for Grad-CAM
    print(f"\nAnalyzing ResNet model for Grad-CAM compatibility...")
    print(f"ResNet model type: {type(resnet_model).__name__}")
    
    # Look for suitable convolutional layers
    conv_layers = []
    for i, layer in enumerate(resnet_model.layers):
        if ('conv' in layer.name.lower() and 
            layer.name.endswith('conv') and 
            'block' in layer.name.lower()):
            conv_layers.append((i, layer.name))
    
    print(f"Found {len(conv_layers)} block convolutional layers")
    
    if conv_layers:
        print("Block conv layers (grouped by block):")
        current_block = ""
        for i, (idx, name) in enumerate(conv_layers):
            block_name = "_".join(name.split("_")[:2])  # e.g., conv4_block1
            if block_name != current_block:
                current_block = block_name
                print(f"  {current_block}:")
            print(f"    {name}")
    
    # Test with a preferred layer
    preferred_layers = [
        'conv5_block3_2_conv',
        'conv5_block2_2_conv',
        'conv4_block6_2_conv',
        'conv4_block5_2_conv',
        'conv3_block4_2_conv'
    ]
    
    print(f"\nTesting preferred layers:")
    for layer_name in preferred_layers:
        try:
            layer = resnet_model.get_layer(layer_name)
            print(f"  ✓ {layer_name} - Found")
        except:
            print(f"  ✗ {layer_name} - Not found")
    
    # Find the best available layer
    print(f"\nFinding best available layer for Grad-CAM:")
    available_layers = []
    for layer_name in preferred_layers:
        try:
            layer = resnet_model.get_layer(layer_name)
            available_layers.append(layer_name)
        except:
            pass
    
    if available_layers:
        best_layer = available_layers[0]
        print(f"Best layer for Grad-CAM: {best_layer}")
        
        # Test building gradient model with this layer
        print(f"\nTesting gradient model with {best_layer}...")
        try:
            layer_obj = resnet_model.get_layer(best_layer)
            
            # Build gradient model
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[layer_obj.output, model.output]
            )
            print("✓ Gradient model built successfully")
            
            # Test gradient computation
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(test_input)
                loss = predictions[0] if len(predictions.shape) == 1 else predictions[:, 0]
            
            grads = tape.gradient(loss, conv_outputs)
            print("✓ Gradient computation successful")
            print("🎉 Grad-CAM should work with this configuration!")
            
        except Exception as e:
            print(f"✗ Error building gradient model: {e}")
    else:
        print("No preferred layers found. Looking for alternatives...")
        
        # Try to find any suitable conv layer
        if conv_layers:
            # Try the last few conv layers
            test_layers = [name for _, name in conv_layers[-5:]]
            print(f"Testing last 5 conv layers: {test_layers}")
            
            for layer_name in test_layers:
                try:
                    layer_obj = resnet_model.get_layer(layer_name)
                    
                    # Build gradient model
                    grad_model = tf.keras.models.Model(
                        inputs=model.input,
                        outputs=[layer_obj.output, model.output]
                    )
                    print(f"✓ Gradient model built successfully with {layer_name}")
                    
                    # Test gradient computation
                    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                    with tf.GradientTape() as tape:
                        conv_outputs, predictions = grad_model(test_input)
                        loss = predictions[0] if len(predictions.shape) == 1 else predictions[:, 0]
                    
                    grads = tape.gradient(loss, conv_outputs)
                    print(f"✓ Gradient computation successful with {layer_name}")
                    print(f"🎉 Grad-CAM should work with {layer_name}!")
                    break
                    
                except Exception as e:
                    print(f"✗ Error with {layer_name}: {e}")
                    continue
            else:
                print("✗ Could not find any working layer for Grad-CAM")

if __name__ == "__main__":
    detailed_model_analysis()