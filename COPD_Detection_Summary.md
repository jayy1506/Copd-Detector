# COPD Detection Using Deep Learning - Summary

## Project Overview
This project implements a deep learning model for early Chronic Obstructive Pulmonary Disease (COPD) detection using chest X-ray images. The model uses transfer learning with ResNet50 architecture to classify images into two categories:
1. Normal (healthy lungs)
2. COPD (including COVID-19 pneumonia and emphysema)

## Implementation Details

### Dataset Structure
The dataset consists of chest X-ray images organized in three directories:
- `train/` - Training images (~5,400 images)
- `val/` - Validation images (~600 images)
- `test/` - Test images (~850 images)

Images are categorized by filename prefixes:
- `Normal*.jpg` - Healthy lung images
- `COVID*.jpg` and `Emphysema*.jpg` - COPD condition images

### Model Architecture
- **Base Model**: ResNet50 with ImageNet pre-trained weights
- **Input Shape**: 224×224×3 (RGB images)
- **Top Layers**: 
  - Global Average Pooling
  - Dense layer with 128 neurons and ReLU activation
  - Dropout layer (0.5)
  - Output sigmoid neuron for binary classification

### Training Process
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy
- **Data Augmentation**: Rotation, width/height shift, horizontal flip, zoom
- **Callbacks**: Early stopping and model checkpointing

### Model Performance
The model has been trained and saved as `best_copd_model.h5` (approximately 95MB).

## Key Features Implemented

### 1. Data Preprocessing
- Automatic loading and labeling based on filename prefixes
- Image resizing to 224×224 pixels
- RGB color space conversion
- Data normalization (pixel values scaled to 0-1)

### 2. Data Augmentation
- Random rotations (±20 degrees)
- Width and height shifts (±20%)
- Horizontal flipping
- Zoom variations (±20%)

### 3. Model Training
- Transfer learning with frozen ResNet50 base layers
- Custom top layers for binary classification
- Early stopping to prevent overfitting
- Best model checkpointing based on validation accuracy

### 4. Model Evaluation
- Accuracy calculation
- Confusion matrix generation
- Classification report (precision, recall, F1-score)
- ROC curve and AUC score

### 5. Explainability with Grad-CAM
- Implementation of Gradient-weighted Class Activation Mapping
- Visualization of regions that influenced model decisions
- Heatmap overlay on original X-ray images

## Medical Disclaimer
**This model is intended for early screening and research purposes only and not for clinical diagnosis.**

## How to Use

### Prerequisites
Install required packages:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn pandas numpy seaborn
```

### Running the Model
1. Ensure the dataset is organized in the required directory structure
2. Run the complete pipeline:
```python
python copd_detection_model.py
```

### Evaluating a Trained Model
To evaluate the saved model:
```python
from tensorflow import keras
model = keras.models.load_model('best_copd_model.h5')
# Use model for predictions
```

## Results
The model demonstrates effective performance in distinguishing between normal and COPD-affected lung X-rays. The Grad-CAM visualizations help understand which regions of the X-ray images the model focuses on when making predictions, providing valuable insights for medical professionals.

## Future Improvements
1. Implement more advanced architectures (EfficientNet, Vision Transformers)
2. Add more comprehensive evaluation metrics
3. Implement cross-validation for more robust performance assessment
4. Add support for additional COPD subtypes
5. Improve the user interface for easier deployment in medical settings