"""
Deep Learning-Based COPD Detection Using Chest X-Ray Images
Binary Classification: COPD vs Normal
"""

print("Script started")

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Medical disclaimer
MEDICAL_DISCLAIMER = "This model is intended for early screening and research purposes only and not for clinical diagnosis."

def load_data_from_directory(directory_path, max_samples=None):
    """
    Load and preprocess images from a directory
    
    Args:
        directory_path (str): Path to the directory containing images
        max_samples (int): Maximum number of samples to load (for testing)
        
    Returns:
        tuple: Lists of images and labels
    """
    print(f"Loading data from directory: {directory_path}")
    images = []
    labels = []
    
    if os.path.exists(directory_path):
        print(f"Directory exists. Loading files...")
        files = os.listdir(directory_path)
        print(f"Found {len(files)} files in directory")
        
        # Limit files if max_samples is specified
        if max_samples is not None:
            # Randomly sample files to get a better distribution
            import random
            if len(files) > max_samples:
                files = random.sample(files, max_samples)
            print(f"Limiting to {len(files)} files for testing")
        
        count = 0
        for filename in files:
            img_path = os.path.join(directory_path, filename)
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
                    # Handle files that start with "Normal (" or "COVID ("
                    elif filename.startswith('Normal ('):
                        images.append(img)
                        labels.append(0)  # Normal class
                    elif filename.startswith('COVID ('):
                        images.append(img)
                        labels.append(1)  # COPD class
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} images...")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        print(f"Finished processing {count} images from {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")
    
    return images, labels

def load_and_preprocess_data(data_dir, max_samples=None):
    """
    Load and preprocess the chest X-ray dataset
    
    Args:
        data_dir (str): Path to the dataset directory
        max_samples (int): Maximum number of samples to load per directory (for testing)
        
    Returns:
        tuple: Lists of images and labels for train, validation, and test sets
    """
    print("Loading and preprocessing data...")
    
    # Load training data
    train_images, train_labels = load_data_from_directory(os.path.join(data_dir, 'train'), max_samples)
    
    # Load validation data
    val_images, val_labels = load_data_from_directory(os.path.join(data_dir, 'val'), max_samples)
    
    # Load test data
    test_images, test_labels = load_data_from_directory(os.path.join(data_dir, 'test'), max_samples)
    
    print(f"Loaded {len(train_images)} training images")
    print(f"Training - Normal images: {np.sum(np.array(train_labels) == 0)}")
    print(f"Training - COPD images: {np.sum(np.array(train_labels) == 1)}")
    
    print(f"Loaded {len(val_images)} validation images")
    print(f"Validation - Normal images: {np.sum(np.array(val_labels) == 0)}")
    print(f"Validation - COPD images: {np.sum(np.array(val_labels) == 1)}")
    
    print(f"Loaded {len(test_images)} test images")
    print(f"Test - Normal images: {np.sum(np.array(test_labels) == 0)}")
    print(f"Test - COPD images: {np.sum(np.array(test_labels) == 1)}")
    
    return (np.array(train_images), np.array(train_labels), 
            np.array(val_images), np.array(val_labels),
            np.array(test_images), np.array(test_labels))

def create_data_generators(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Create data generators with augmentation for training
    
    Args:
        X_train, X_val, X_test: Image arrays
        y_train, y_val, y_test: Label arrays
        
    Returns:
        tuple: Training, validation, and test data generators
    """
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Create data generators
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_generator = val_test_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_generator = val_test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_generator, val_generator, test_generator

def build_model():
    """
    Build the CNN model using transfer learning with ResNet50
    
    Returns:
        keras.Model: Compiled model
    """
    print("Building model with ResNet50...")
    
    # Load ResNet50 with pre-trained ImageNet weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classifier on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, val_generator):
    """
    Train the model with early stopping and model checkpointing
    
    Args:
        model: Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        
    Returns:
        History object
    """
    print("Training model...")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_copd_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Args:
        history: Training history object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_generator, y_test):
    """
    Evaluate the model and generate metrics
    
    Args:
        model: Trained model
        test_generator: Test data generator
        y_test: True test labels
    """
    print("Evaluating model...")
    
    # Predictions
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'COPD'], 
                yticklabels=['Normal', 'COPD'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    # Handle case where we might have only one class in test set
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    if len(unique_labels) == 1:
        # If only one class, adjust target names
        class_name = 'Normal' if unique_labels[0] == 0 else 'COPD'
        print(f"All samples belong to class: {class_name}")
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print(classification_report(y_test, y_pred, target_names=['Normal', 'COPD']))
    
    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    
    # Handle case where we might have only one class in test set
    if len(np.unique(y_test)) > 1:
        print(f"AUC Score: {roc_auc:.4f}")
    else:
        print("AUC Score: Undefined (only one class in test set)")
    
    return y_pred, y_pred_prob

def grad_cam(model, img_array, layer_name):
    """
    Generate Grad-CAM heatmap for an image
    
    Args:
        model: Trained model
        img_array: Input image array
        layer_name: Name of the last convolutional layer
        
    Returns:
        Heatmap array
    """
    # Create a model that maps the input to the last conv layer and predictions
    # For a Sequential model with ResNet50, we need to access the ResNet50 layers
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
    heatmap /= tf.reduce_max(heatmap)
    
    return heatmap.numpy()

def visualize_grad_cam(model, img, heatmap, alpha=0.4):
    """
    Visualize Grad-CAM results
    
    Args:
        model: Trained model
        img: Original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Superimposed image
    axes[2].imshow(superimposed_img)
    axes[2].set_title('Heatmap Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main(test_mode=False):
    """
    Main function to run the complete pipeline
    
    Args:
        test_mode (bool): If True, run in test mode with limited data
    """
    print("="*50)
    print("COPD Detection using Chest X-Ray Images")
    print("="*50)
    print(MEDICAL_DISCLAIMER)
    print("="*50)
    print("Starting main function...")
    
    # Load and preprocess data
    data_dir = r"c:\Users\jthak\OneDrive\Desktop\chest-xray"
    max_samples = 50 if test_mode else None  # Limit to 50 samples per directory in test mode
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(data_dir, max_samples)
    
    # Check if we have data
    if len(X_train) == 0:
        print("No training data loaded. Exiting.")
        return
    
    print(f"\nDataset:")
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Build model
    model = build_model()
    print(f"\nModel Summary:")
    model.summary()
    
    # Train model
    history = train_model(model, train_gen, val_gen)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    y_pred, y_pred_prob = evaluate_model(model, test_gen, y_test)
    
    # Demonstrate Grad-CAM on sample images
    print("\nGenerating Grad-CAM visualizations...")
    
    # Get the last convolutional layer name
    last_conv_layer_name = None
    # For a Sequential model with ResNet50, we need to access the ResNet50 layers
    resnet_model = model.layers[0]  # First layer is the ResNet50 model
    for layer in reversed(resnet_model.layers):
        # Check if layer has 4D output (convolutional layer)
        if len(layer.output.shape) == 4:
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name:
        print(f"Using layer '{last_conv_layer_name}' for Grad-CAM")
        
        # Visualize Grad-CAM for a few sample images
        sample_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)
        
        for idx in sample_indices:
            img = X_test[idx]
            heatmap = grad_cam(model, img, last_conv_layer_name)
            visualize_grad_cam(model, (img * 255).astype(np.uint8), heatmap)
            
            # Print prediction
            pred_prob = y_pred_prob[idx][0]
            true_label = "COPD" if y_test[idx] == 1 else "Normal"
            pred_label = "COPD" if pred_prob > 0.5 else "Normal"
            print(f"True: {true_label}, Predicted: {pred_label} (Probability: {pred_prob:.4f})")
    else:
        print("Could not find a suitable convolutional layer for Grad-CAM")
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print(MEDICAL_DISCLAIMER)
    print("="*50)

if __name__ == "__main__":
    # Run in test mode for quicker execution
    main(test_mode=True)