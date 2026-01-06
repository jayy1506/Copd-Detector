import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def test_model_loading_and_prediction():
    # Load the trained model
    model_path = 'best_copd_model_trained.h5'
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Get a sample image from the test directory
    test_dir = r"C:\Users\jthak\OneDrive\Attachments\Desktop\chest-xray\test"
    image_files = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(file)
    
    if image_files:
        sample_image = image_files[0]  # Get the first image
        sample_path = os.path.join(test_dir, sample_image)
        
        print(f"Testing prediction on sample image: {sample_image}")
        
        # Load and preprocess the image
        img = load_img(sample_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        prediction_prob = prediction[0][0]
        prediction_label = "COPD" if prediction_prob > 0.5 else "Normal"
        
        print(f"Prediction probability: {prediction_prob:.4f}")
        print(f"Prediction: {prediction_label}")
        
        # Determine actual class from filename
        filename = os.path.basename(sample_image).lower()
        actual_label = "COPD" if not filename.startswith('normal') else "Normal"
        print(f"Actual: {actual_label}")
        print(f"Correct: {prediction_label == actual_label}")
    
    print("\nModel test completed successfully!")

if __name__ == "__main__":
    test_model_loading_and_prediction()