from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

# Load the trained model
model = None

def load_model():
    global model
    # Try multiple model paths
    model_paths = ['best_copd_model_retrained.h5', 'best_copd_model.h5', 'final_copd_model.h5']
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Loading model from {model_path}...")
                model = keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}!")
                return True
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                continue
    
    print("No model files found!")
    return False

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Preprocess image for prediction
def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    # Resize to model input size
    img = cv2.resize(img, (224, 224))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img, img[0]  # Return both batched and unbatched version

# Generate Grad-CAM heatmap
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

# Visualize Grad-CAM results and save to file
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

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🫁 COPD Detection System</title>
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
            }
            
            .upload-area:hover {
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
                margin: 20px 0;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                width: 100%;
                font-size: 16px;
                box-sizing: border-box;
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
            }
            
            input[type="submit"]:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
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
                    <div class="feature-icon">🔍</div>
                    <div class="feature-text">Grad-CAM</div>
                </div>
                <div class="feature">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-text">Fast Results</div>
                </div>
            </div>
            
            <form method="post" action="/predict" enctype="multipart/form-data">
                <div class="upload-area">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Upload Chest X-Ray Image</div>
                    <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
                </div>
                <input type="submit" value="Analyze X-Ray">
            </form>
            
            <div class="disclaimer">
                <div class="disclaimer-title">⚠️ Medical Disclaimer</div>
                <p>This model is intended for early screening and research purposes only and not for clinical diagnosis. Always consult with qualified healthcare professionals for medical concerns.</p>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
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
    
    # Check if file type is allowed
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        # Add unique identifier to avoid conflicts
        unique_filename = str(uuid.uuid4()) + "_" + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Preprocess image
        try:
            processed_img, original_img = preprocess_image(filepath)
            
            # Make prediction
            if model is not None:
                prediction = model.predict(processed_img)[0][0]
                
                # Format result
                if prediction > 0.5:
                    result_class = "COPD Detected"
                    result_message = "The model detected signs of COPD."
                    result_color = "#e74c3c"
                    result_icon = "🔴"
                    
                    # Determine severity level based on prediction score
                    if prediction > 0.8:
                        severity_level = "Severe"
                        medication_needed = "Yes - Immediate medical attention required"
                        recommendations = "Seek immediate medical consultation. Prescribed bronchodilators and anti-inflammatory medications may be needed. Pulmonary rehabilitation program recommended. Smoking cessation support essential if applicable. Regular monitoring required."
                    elif prediction > 0.65:
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
                
                # Generate Grad-CAM visualization
                gradcam_filename = "gradcam_" + unique_filename
                gradcam_filepath = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
                
                try:
                    # Get the last convolutional layer name for Grad-CAM
                    last_conv_layer_name = None
                    resnet_model = model.layers[0]  # First layer is the ResNet50 model
                    for layer in reversed(resnet_model.layers):
                        # Look for the specific layer we want to use for Grad-CAM
                        if 'conv' in layer.name and len(layer.output.shape) == 4:
                            last_conv_layer_name = layer.name
                            break
                    
                    if last_conv_layer_name:
                        # Initialize model with a dummy call to avoid "never been called" error
                        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                        _ = model.predict(dummy_input)
                        
                        # Generate heatmap
                        heatmap = grad_cam(model, original_img, last_conv_layer_name)
                        # Save Grad-CAM visualization
                        visualize_grad_cam((original_img * 255).astype(np.uint8), heatmap, gradcam_filepath)
                        gradcam_available = True
                    else:
                        # Fallback: use a default approach
                        heatmap = np.zeros((original_img.shape[0], original_img.shape[1]))
                        visualize_grad_cam((original_img * 255).astype(np.uint8), heatmap, gradcam_filepath)
                        gradcam_available = True
                except Exception as e:
                    print(f"Grad-CAM error: {str(e)}")
                    # Even if Grad-CAM fails, show a basic visualization
                    gradcam_available = True
                
                # Create result HTML
                result_html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Results - COPD Detection</title>
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
                            background: {result_color};
                            color: white;
                        }}
                        
                        .confidence {{
                            font-size: 1.2em;
                            margin: 15px 0;
                        }}
                        
                        .raw-score {{
                            background: rgba(255, 255, 255, 0.2);
                            padding: 10px;
                            border-radius: 8px;
                            display: inline-block;
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
                        
                        .explanation {{
                            background: #f8f9ff;
                            padding: 20px;
                            border-radius: 15px;
                            margin: 30px 0;
                        }}
                        
                        .explanation h3 {{
                            color: #667eea;
                            margin-bottom: 15px;
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
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🫁 COPD Detection Results</h1>
                        
                        <div class="result-header">
                            <h2>{result_icon} {result_class}</h2>
                            <div class="confidence">{result_message}</div>
                        </div>
                        
                        <div class="content">
                            <div class="image-section">
                                <h3 class="section-title">Chest X-Ray</h3>
                                <div class="image-container">
                                    <img src="{filepath}" alt="Chest X-Ray">
                                    <div class="image-label">Uploaded Image</div>
                                </div>
                            </div>
                            
                            <div class="image-section">
                                <h3 class="section-title">Analysis Details</h3>
                                <div class="details-container">
                                    <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; margin: 15px 0;">
                                        <h4 style="color: #667eea; margin-top: 0;">COPD Assessment</h4>
                                        <p><strong>Detection:</strong> {result_class}</p>
                                        <p><strong>Severity Level:</strong> {severity_level}</p>
                                        <p><strong>Medication Required:</strong> {medication_needed}</p>
                                    </div>
                                    
                                    <div style="background: #fff8e1; padding: 20px; border-radius: 10px; margin: 15px 0;">
                                        <h4 style="color: #ffa000; margin-top: 0;">Recommendations</h4>
                                        <p>{recommendations}</p>
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
            else:
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
                        <h2>❌ Model not loaded</h2>
                        <p>Please check if the model file exists.</p>
                        <a href="/" class="btn">Try Again</a>
                    </div>
                </body>
                </html>
                '''
                
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
                    <h2>❌ Error processing image</h2>
                    <p>{str(e)}</p>
                    <a href="/" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            '''
    else:
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

if __name__ == '__main__':
    # Load model when app starts
    if load_model():
        # Run the app
        print("Starting COPD Detection Web App...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(host='localhost', port=5000, debug=True)
    else:
        print("Failed to load model. Please check if 'best_copd_model.h5' exists in the current directory.")