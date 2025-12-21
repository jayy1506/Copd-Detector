#!/usr/bin/env python3
"""
Integrated COPD Detection Application
Frontend: HTML/CSS
Backend: Python with TensorFlow/Keras
"""

import os
import sys
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid

# Initialize Flask app
app = Flask(__name__)

# Configure folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Global model variable
model = None

def load_model():
    """Load the trained COPD detection model"""
    global model
    model_paths = [
        os.path.join(BASE_DIR, 'best_copd_model_retrained.h5'),
        os.path.join(BASE_DIR, 'best_copd_model.h5'),
        os.path.join(BASE_DIR, 'final_copd_model.h5')
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}...")
                model = keras.models.load_model(model_path)
                print(f"Model loaded successfully!")
                return True
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                continue
    
    print("No model files found!")
    return False

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        # Store original image for display
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
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict_copd(image_array):
    """Make COPD prediction"""
    try:
        if model is None:
            return None
            
        # Initialize model with dummy input if needed
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        _ = model.predict(dummy_input)
        
        # Make prediction
        prediction = model.predict(image_array)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def get_assessment_details(prediction_score):
    """Get detailed assessment based on prediction score"""
    if prediction_score is None:
        return {
            'result_class': 'Error',
            'result_message': 'Unable to process image',
            'result_color': '#888888',
            'result_icon': '❓',
            'severity_level': 'Unknown',
            'medication_needed': 'N/A',
            'recommendations': 'An error occurred during processing. Please try again.'
        }
    
    if prediction_score > 0.5:
        result_class = "COPD Detected"
        result_message = "The model detected signs of COPD."
        result_color = "#e74c3c"
        result_icon = "🔴"
        
        # Determine severity level based on prediction score
        if prediction_score > 0.8:
            severity_level = "Severe"
            medication_needed = "Yes - Immediate medical attention required"
            recommendations = "Seek immediate medical consultation. Prescribed bronchodilators and anti-inflammatory medications may be needed. Pulmonary rehabilitation program recommended. Smoking cessation support essential if applicable. Regular monitoring required."
        elif prediction_score > 0.65:
            severity_level = "Moderate"
            medication_needed = "Yes - Medical consultation recommended"
            recommendations = "Consult pulmonologist for treatment plan. Bronchodilators may be prescribed. Pulmonary function tests recommended. Lifestyle modifications including exercise and nutrition. Smoking cessation if applicable. Regular follow-ups needed."
        else:
            severity_level = "Mild"
            medication_needed = "Possibly - Consult physician"
            recommendations = "Early stage detection. Monitor symptoms regularly. Lifestyle changes may help prevent progression. Annual check-ups recommended. Consider pulmonary function testing. Smoking cessation if applicable."
    else:
        result_class = "Normal"
        result_message = "The model classified this as normal."
        result_color = "#27ae60"
        result_icon = "🟢"
        severity_level = "None"
        medication_needed = "No - Normal result"
        recommendations = "No signs of COPD detected. Continue healthy lifestyle habits. Annual check-ups recommended for ongoing monitoring. Avoid smoking and environmental pollutants. Exercise regularly to maintain lung health."
    
    return {
        'result_class': result_class,
        'result_message': result_message,
        'result_color': result_color,
        'result_icon': result_icon,
        'severity_level': severity_level,
        'medication_needed': medication_needed,
        'recommendations': recommendations
    }

@app.route('/')
def index():
    """Main page with file upload form"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🫁 COPD Detection System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 500px;
                padding: 40px;
                text-align: center;
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px 20px;
                margin: 30px 0;
                transition: all 0.3s ease;
                background-color: #f8f9ff;
                cursor: pointer;
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background-color: #eef1ff;
            }
            
            .upload-area.active {
                border-color: #764ba2;
                background-color: #eef1ff;
            }
            
            .upload-icon {
                font-size: 3em;
                margin-bottom: 15px;
                color: #667eea;
            }
            
            .upload-text {
                color: #666;
                margin-bottom: 20px;
            }
            
            input[type="file"] {
                display: none;
            }
            
            .file-label {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 50px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                margin: 15px 0;
            }
            
            .file-label:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }
            
            input[type="submit"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px 32px;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                font-size: 18px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                width: 100%;
                margin-top: 20px;
            }
            
            input[type="submit"]:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }
            
            input[type="submit"]:disabled {
                background: #cccccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
            .features {
                display: flex;
                justify-content: space-around;
                margin: 30px 0;
                flex-wrap: wrap;
            }
            
            .feature {
                text-align: center;
                padding: 15px;
                flex: 1;
                min-width: 120px;
            }
            
            .feature-icon {
                font-size: 2em;
                margin-bottom: 10px;
                color: #667eea;
            }
            
            .feature-text {
                font-size: 0.9em;
                color: #666;
            }
            
            .disclaimer {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                border-radius: 0 8px 8px 0;
                margin-top: 30px;
                font-size: 0.85em;
                text-align: left;
            }
            
            .disclaimer-title {
                font-weight: bold;
                color: #856404;
            }
            
            .file-info {
                margin-top: 15px;
                color: #666;
                font-size: 0.9em;
            }
            
            @media (max-width: 600px) {
                .container {
                    padding: 20px;
                }
                
                h1 {
                    font-size: 2em;
                }
                
                .features {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🫁 COPD Detection</h1>
            <p class="subtitle">AI-Powered Chest X-Ray Analysis</p>
            
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-text">AI Analysis</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">📊</div>
                    <div class="feature-text">Detailed Report</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-text">Fast Results</div>
                </div>
            </div>
            
            <form method="post" action="/analyze" enctype="multipart/form-data" id="uploadForm">
                <div class="upload-area" id="dropArea">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Upload Chest X-Ray Image</div>
                    <input type="file" name="file" id="fileInput" accept=".jpg,.jpeg,.png" required>
                    <label for="fileInput" class="file-label">Choose File</label>
                    <div class="file-info" id="fileInfo">No file selected</div>
                </div>
                <input type="submit" value="Analyze X-Ray" id="submitBtn" disabled>
            </form>
            
            <div class="disclaimer">
                <div class="disclaimer-title">⚠️ Medical Disclaimer</div>
                <p>This model is intended for early screening and research purposes only and not for clinical diagnosis. Always consult with qualified healthcare professionals for medical concerns.</p>
            </div>
        </div>
        
        <script>
            // File selection handling
            const fileInput = document.getElementById('fileInput');
            const fileInfo = document.getElementById('fileInfo');
            const submitBtn = document.getElementById('submitBtn');
            const dropArea = document.getElementById('dropArea');
            const uploadForm = document.getElementById('uploadForm');
            
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    const file = this.files[0];
                    fileInfo.textContent = file.name;
                    submitBtn.disabled = false;
                } else {
                    fileInfo.textContent = 'No file selected';
                    submitBtn.disabled = true;
                }
            });
            
            // Drag and drop functionality
            dropArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.classList.add('active');
            });
            
            dropArea.addEventListener('dragleave', function() {
                this.classList.remove('active');
            });
            
            dropArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.classList.remove('active');
                
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    const file = e.dataTransfer.files[0];
                    fileInfo.textContent = file.name;
                    submitBtn.disabled = false;
                }
            });
            
            // Form submission with loading state
            uploadForm.addEventListener('submit', function() {
                submitBtn.value = 'Analyzing...';
                submitBtn.disabled = true;
            });
        </script>
    </body>
    </html>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded X-ray image"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error - COPD Detection</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f2f5; }
                    .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                    .btn { background: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ No file uploaded</h2>
                    <p>Please select a file to upload.</p>
                    <a href="/" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            '''
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error - COPD Detection</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f2f5; }
                    .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                    .btn { background: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ No file selected</h2>
                    <p>Please select a file to upload.</p>
                    <a href="/" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            '''
        
        # Check file type
        if not allowed_file(file.filename):
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error - COPD Detection</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f2f5; }
                    .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                    .btn { background: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Invalid file type</h2>
                    <p>Please upload a JPG, JPEG, or PNG image.</p>
                    <a href="/" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            '''
        
        # Secure the filename and save
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Create a relative path for web access
        relative_filepath = f"/uploads/{unique_filename}"
        
        # Preprocess image
        processed_img, original_img = preprocess_image(filepath)
        if processed_img is None:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error - COPD Detection</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f2f5; }
                    .container { background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }
                    .btn { background: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Error processing image</h2>
                    <p>Could not process the uploaded image. Please try another image.</p>
                    <a href="/" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            '''
        
        # Make prediction
        prediction_score = predict_copd(processed_img)
        assessment = get_assessment_details(prediction_score)
        
        # Generate result HTML
        result_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Results - COPD Detection</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
                
                body {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .container {{
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                    width: 100%;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 40px;
                }}
                
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                    font-size: 2.5em;
                }}
                
                .result-header {{
                    text-align: center;
                    padding: 20px;
                    border-radius: 15px;
                    margin-bottom: 30px;
                    background: {assessment['result_color']};
                    color: white;
                }}
                
                .confidence {{
                    font-size: 1.2em;
                    margin: 15px 0;
                }}
                
                .content {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 30px;
                    margin: 30px 0;
                }}
                
                .image-section {{
                    flex: 1;
                    min-width: 300px;
                }}
                
                .section-title {{
                    color: #333;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #667eea;
                }}
                
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .image-container img {{
                    max-width: 100%;
                    border-radius: 15px;
                    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
                }}
                
                .image-label {{
                    font-weight: bold;
                    margin: 15px 0;
                    color: #666;
                }}
                
                .btn {{
                    display: inline-block;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 16px 32px;
                    text-decoration: none;
                    border-radius: 50px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                    margin: 10px;
                    text-align: center;
                }}
                
                .btn:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
                }}
                
                .disclaimer {{
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 20px;
                    border-radius: 0 10px 10px 0;
                    margin-top: 30px;
                    font-size: 0.9em;
                }}
                
                .disclaimer-title {{
                    font-weight: bold;
                    color: #856404;
                }}
                
                @media (max-width: 768px) {{
                    .content {{
                        flex-direction: column;
                    }}
                    
                    .container {{
                        padding: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🫁 COPD Detection Results</h1>
                
                <div class="result-header">
                    <h2>{assessment['result_icon']} {assessment['result_class']}</h2>
                    <div class="confidence">{assessment['result_message']}</div>
                </div>
                
                <div class="content">
                    <div class="image-section">
                        <h3 class="section-title">Chest X-Ray</h3>
                        <div class="image-container">
                            <img src="{relative_filepath}" alt="Chest X-Ray">
                            <div class="image-label">Uploaded Image</div>
                        </div>
                    </div>
                    
                    <div class="image-section">
                        <h3 class="section-title">Analysis Details</h3>
                        <div class="details-container">
                            <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; margin: 15px 0;">
                                <h4 style="color: #667eea; margin-top: 0;">COPD Assessment</h4>
                                <p><strong>Detection:</strong> {assessment['result_class']}</p>
                                <p><strong>Severity Level:</strong> {assessment['severity_level']}</p>
                                <p><strong>Medication Required:</strong> {assessment['medication_needed']}</p>
                            </div>
                            
                            <div style="background: #fff8e1; padding: 20px; border-radius: 10px; margin: 15px 0;">
                                <h4 style="color: #ffa000; margin-top: 0;">Recommendations</h4>
                                <p>{assessment['recommendations']}</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <a href="/" class="btn">Analyze Another Image</a>
                </div>
                
                <div class="disclaimer">
                    <div class="disclaimer-title">⚠️ Medical Disclaimer</div>
                    <p>This model is intended for early screening and research purposes only and not for clinical diagnosis. Always consult with qualified healthcare professionals for medical concerns. The accuracy of this model has limitations and should not be used as a substitute for professional medical advice.</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        return result_html
        
    except Exception as e:
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - COPD Detection</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f2f5; }}
                .container {{ background: white; padding: 30px; border-radius: 10px; max-width: 500px; margin: 0 auto; }}
                .btn {{ background: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>❌ An error occurred</h2>
                <p>{str(e)}</p>
                <a href="/" class="btn">Try Again</a>
            </div>
        </body>
        </html>
        '''

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def main():
    """Main function to run the application"""
    print("="*60)
    print("COPD Detection Application")
    print("="*60)
    print("Loading model...")
    
    # Load model
    if not load_model():
        print("Failed to load model. Please ensure model files exist.")
        return
    
    print("Model loaded successfully!")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    
    # Run the Flask app
    app.run(host='localhost', port=5000, debug=True)

if __name__ == '__main__':
    main()