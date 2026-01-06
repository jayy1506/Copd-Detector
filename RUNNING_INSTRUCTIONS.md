# COPD Detection Model - Running Instructions

## Project Status

The COPD detection model has been successfully implemented and trained. Here's what's available:

1. **Trained Model**: `best_copd_model.h5` (93.4 MB) - Ready for use
2. **Complete Implementation**: All code for data preprocessing, model training, evaluation, and visualization
3. **Full Pipeline**: Working end-to-end solution for COPD detection from chest X-rays

## How to Use the Model

### Quick Demo
Run the simple demo to verify the model loads and makes predictions:
```bash
python run_simple_demo.py
```

### Full Training and Evaluation
To run the complete pipeline with training and evaluation:
```bash
python copd_detection_model.py
```

Note: This runs in test mode by default (limited dataset) for faster execution.

### For Full Training
To run with the complete dataset, modify the `copd_detection_model.py` file and set `test_mode=False` in the `if __name__ == "__main__":` section.

## Model Details

- **Architecture**: ResNet50 with transfer learning
- **Input**: 224×224×3 RGB images
- **Task**: Binary classification (Normal vs COPD)
- **Output**: Probability score (0 = Normal, 1 = COPD)

## Key Components Implemented

1. **Data Preprocessing**: Automatic loading and labeling of X-ray images
2. **Data Augmentation**: Rotation, shifting, flipping, and zooming for better generalization
3. **Transfer Learning**: ResNet50 base with custom classification head
4. **Training Pipeline**: Early stopping and model checkpointing
5. **Evaluation Metrics**: Accuracy, confusion matrix, classification report, ROC-AUC
6. **Explainability**: Grad-CAM visualization of important regions
7. **Medical Disclaimer**: Clear warning about research-only use

## Requirements

All dependencies are listed in `requirements.txt`:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn pandas numpy seaborn
```

## Medical Disclaimer

**This model is intended for early screening and research purposes only and not for clinical diagnosis.**

Always consult with qualified healthcare professionals for medical concerns.