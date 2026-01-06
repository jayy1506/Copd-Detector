#!/usr/bin/env python3
"""
Train the COPD detection model with specified data paths
This script trains a model using the train and test directories you specified.
"""

import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
import random

# Set random seeds for reproducibility
np.random.seed(42)

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

def load_and_preprocess_data(train_dir, test_dir, val_dir=None, max_samples=None):
    """
    Load and preprocess the chest X-ray dataset from specified directories
    
    Args:
        train_dir (str): Path to the training directory
        test_dir (str): Path to the test directory
        val_dir (str): Path to the validation directory (optional)
        max_samples (int): Maximum number of samples to load per directory (for testing)
        
    Returns:
        tuple: Lists of images and labels for train, validation, and test sets
    """
    print("Loading and preprocessing data...")
    
    # Load training data
    train_images, train_labels = load_data_from_directory(train_dir, max_samples)
    
    # Load validation data if directory exists, otherwise create from training data
    if val_dir and os.path.exists(val_dir):
        val_images, val_labels = load_data_from_directory(val_dir, max_samples)
    else:
        print("Validation directory not provided, creating validation set from training data...")
        # Create a validation set by splitting training data (20% for validation)
        split_idx = int(0.8 * len(train_images))
        val_images = train_images[split_idx:]
        val_labels = train_labels[split_idx:]
        train_images = train_images[:split_idx]
        train_labels = train_labels[:split_idx]
    
    # Load test data
    test_images, test_labels = load_data_from_directory(test_dir, max_samples)
    
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
    
    # Shuffle training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    # Create data generators
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
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
    base_model = keras.applications.ResNet50(
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
        'best_copd_model_trained.h5',
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

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        
    Returns:
        Test loss and accuracy
    """
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    return test_loss, test_accuracy

def main():
    """
    Main function to run the complete pipeline
    """
    print("="*50)
    print("COPD Detection using Chest X-Ray Images - TRAINING")
    print("="*50)
    print(MEDICAL_DISCLAIMER)
    print("="*50)
    print("Starting training process...")
    
    # Define the data directories as per your request
    train_dir = r"C:\Users\jthak\OneDrive\Attachments\Desktop\chest-xray\train"
    test_dir = r"C:\Users\jthak\OneDrive\Attachments\Desktop\chest-xray\test"
    val_dir = r"C:\Users\jthak\OneDrive\Attachments\Desktop\chest-xray\val"  # Using val if available
    
    print(f"Training directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Validation directory: {val_dir}")
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(
        train_dir, test_dir, val_dir, max_samples=None  # Using all available data
    )
    
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
    
    # Evaluate model on test data
    test_loss, test_accuracy = evaluate_model(model, test_gen)
    
    # Save final model
    model.save('final_copd_model_trained.h5')
    print("Model saved as 'final_copd_model_trained.h5'")
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(MEDICAL_DISCLAIMER)
    print("="*50)

if __name__ == "__main__":
    import tensorflow as tf
    main()