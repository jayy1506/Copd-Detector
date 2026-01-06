import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_evaluate_model():
    # Load the trained model
    model_path = 'best_copd_model_trained.h5'
    if os.path.exists(model_path):
        print("Loading best trained model...")
        model = tf.keras.models.load_model(model_path)
    else:
        model_path = 'final_copd_model.h5'
        print("Loading final trained model...")
        model = tf.keras.models.load_model(model_path)
    
    # Define paths
    test_dir = r"C:\Users\jthak\OneDrive\Attachments\Desktop\chest-xray\test"
    
    # Get all image files in test directory
    image_files = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(file)
    
    print(f"Found {len(image_files)} test images")
    
    # Prepare to store predictions and true labels
    predictions = []
    true_labels = []
    
    # Process each image
    for i, file in enumerate(image_files):
        # Load and preprocess image
        img_path = os.path.join(test_dir, file)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        
        # Make prediction
        pred = model.predict(img_array, verbose=0)
        predictions.append(pred[0][0])  # Store the probability
        
        # Determine true label based on filename (like in training)
        filename = os.path.basename(file).lower()
        if filename.startswith('normal'):
            true_labels.append(0)  # Normal class
        else:
            true_labels.append(1)  # COPD class (COVID/Emphysema)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Convert predictions to binary (0 or 1) using 0.5 threshold
    predicted_labels = (predictions > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    class_labels = ['Normal', 'COPD']
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_labels))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_test.png')
    plt.show()
    
    print("Confusion Matrix saved as confusion_matrix_test.png")
    
    # Calculate additional metrics
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))
    true_negatives = np.sum((true_labels == 0) & (predicted_labels == 0))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == 0))
    
    print(f"\nDetailed Results:")
    print(f"True Positives (COPD correctly identified): {true_positives}")
    print(f"False Positives (Normal incorrectly called COPD): {false_positives}")
    print(f"True Negatives (Normal correctly identified): {true_negatives}")
    print(f"False Negatives (COPD incorrectly called Normal): {false_negatives}")
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = load_and_evaluate_model()
    print(f"\nModel evaluation completed! Final test accuracy: {accuracy:.4f}")