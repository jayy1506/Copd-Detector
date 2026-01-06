#!/usr/bin/env python3
"""
Test Grad-CAM functionality
"""

import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf

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
        return None, None
        
    # Store original image for visualization
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (224, 224))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img, original_img

def grad_cam(model, img_array, layer_name):
    """
    Generate Grad-CAM heatmap for an image
    """
    # Create a model that maps the input to the last conv layer and predictions
    resnet_model = model.layers[0]  # First layer is the ResNet50 model
    
    # Build the gradient model correctly
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[resnet_model.get_layer(layer_name).output, model.output]
    )
    
    # Expand dimensions for batch size
    img_array = np.expand_dims(img_array, axis=0)
    
    with tf.GradientTape() as tape:
        # Get the outputs from the gradient model
        conv_outputs, predictions = grad_model(img_array)
        # For binary classification, we use the output directly
        loss = predictions[0] if len(predictions.shape) == 1 else predictions[:, 0]
    
    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted sum of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Apply ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)
    
    return heatmap.numpy()

def visualize_grad_cam(img, heatmap, save_path, alpha=0.4):
    """
    Visualize Grad-CAM results and save to file
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
    
    # Save the result
    cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    return save_path

def find_last_conv_layer(model):
    """Find the last convolutional layer in the ResNet model"""
    resnet_model = model.layers[0]  # First layer is the ResNet50 model
    for layer in reversed(resnet_model.layers):
        # Look for the specific layer we want to use for Grad-CAM
        if 'conv' in layer.name and len(layer.output.shape) == 4:
            print(f"Found last conv layer: {layer.name}")
            return layer.name
    return None

def test_gradcam_with_sample_image():
    """Test Grad-CAM with a sample image from the test set"""
    # Load model
    model, model_path = load_model()
    if model is None:
        return
    
    # Find test images
    test_dir = './test'
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    files = os.listdir(test_dir)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in test directory")
        return
    
    # Take first image for testing
    filename = image_files[0]
    filepath = os.path.join(test_dir, filename)
    print(f"Testing Grad-CAM with image: {filename}")
    
    # Preprocess image
    processed_img, original_img = preprocess_image(filepath)
    if processed_img is None:
        return
    
    # Find last convolutional layer
    last_conv_layer_name = find_last_conv_layer(model)
    if not last_conv_layer_name:
        print("Could not find last convolutional layer")
        return
    
    # Generate heatmap
    try:
        heatmap = grad_cam(model, original_img, last_conv_layer_name)
        print(f"Heatmap generated successfully. Shape: {heatmap.shape}")
        print(f"Heatmap min: {np.min(heatmap)}, max: {np.max(heatmap)}")
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return
    
    # Save Grad-CAM visualization
    save_path = f"gradcam_test_{filename}"
    try:
        visualize_grad_cam(original_img, heatmap, save_path)
        print(f"Grad-CAM visualization saved to: {save_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

def main():
    print("Grad-CAM Test Tool")
    print("=" * 30)
    test_gradcam_with_sample_image()

if __name__ == "__main__":
    main()