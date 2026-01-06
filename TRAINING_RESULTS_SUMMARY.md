# COPD Detection Model - Training Summary

## Model Information
- Model: ResNet50-based binary classifier for COPD detection
- Task: Binary classification (Normal vs COPD)
- Architecture: Transfer learning with ResNet50 base

## Training Data
- Training images: 5,436 (2,671 Normal, 2,765 COPD)
- Validation images: 601 (300 Normal, 301 COPD) 
- Test images: 850 (300 Normal, 550 COPD)

## Training Process
- Training completed for all 20 epochs
- Best model saved as: `best_copd_model_trained.h5`
- Final model saved as: `final_copd_model.h5`
- Training accuracy improved from ~59% to ~90%+ over epochs
- Validation accuracy remained consistently high (~91%)

## Test Results
- **Overall Test Accuracy: 92.24%**
- Normal class precision: 84%, recall: 97%, F1-score: 90%
- COPD class precision: 98%, recall: 90%, F1-score: 94%

## Detailed Performance
- True Positives (COPD correctly identified): 493
- False Positives (Normal incorrectly called COPD): 9
- True Negatives (Normal correctly identified): 291
- False Negatives (COPD incorrectly called Normal): 57

## Model Performance Notes
- The model shows excellent performance with 92.24% overall accuracy
- High precision for COPD class (98%) indicates few false alarms
- Good recall for COPD class (90%) indicates good detection of actual COPD cases
- The model is conservative in predicting COPD, which might be appropriate for medical diagnosis
- Confusion matrix saved as confusion_matrix_test.png

## File Locations
- Trained model: best_copd_model_trained.h5
- Final model: final_copd_model.h5
- Confusion matrix: confusion_matrix_test.png