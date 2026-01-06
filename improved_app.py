from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = None
def load_model():
    global model
    model_path = 'best_copd_model.h5'
    if os.path.exists(model_path):
        print("Loading model...")
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return True
    else:
        print(f"Model file {model_path} not found!")
        return False

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        <title>🫁 COPD Detection Model</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }
            .container { max-width: 800px; margin: 0 auto; padding: 30px; }
            header { text-align: center; padding: 30px 0; }
            h1 { color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; }
            .subtitle { color: #7f8c8d; font-size: 1.2em; margin-bottom: 30px; }
            .card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 30px; }
            .upload-area { border: 3px dashed #3498db; border-radius: 10px; padding: 40px; text-align: center; margin: 30px 0; transition: all 0.3s ease; background-color: #f8f9fa; }
            .upload-area:hover { border-color: #2980b9; background-color: #e3f2fd; }
            .upload-area h2 { color: #3498db; margin-top: 0; }
            input[type="file"] { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; width: 100%; box-sizing: border-box; }
            input[type="submit"] { background: linear-gradient(to right, #3498db, #2980b9); color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3); }
            input[type="submit"]:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4); }
            .disclaimer { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; border-radius: 0 8px 8px 0; margin-top: 30px; font-size: 14px; }
            .disclaimer strong { color: #856404; }
            footer { text-align: center; padding: 20px; color: #7f8c8d; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🫁 COPD Detection Model</h1>
                <div class="subtitle">AI-Powered Early Detection of Chronic Obstructive Pulmonary Disease</div>
            </header>
            
            <div class="card">
                <h2>Upload Chest X-Ray Image</h2>
                <p>This tool uses deep learning to analyze chest X-ray images for early detection of COPD.</p>
                
                <form method="post" action="/predict" enctype="multipart/form-data">
                    <div class="upload-area">
                        <h2>📁 Drop your X-Ray image here</h2>
                        <p>or</p>
                        <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
                    </div>
                    <input type="submit" value="Analyze X-Ray">
                </form>
            </div>
            
            <div class="disclaimer">
                <strong>⚠️ Medical Disclaimer:</strong> This model is intended for early screening and research purposes only and not for clinical diagnosis. Always consult with qualified healthcare professionals for medical concerns. The accuracy of this model has limitations and should not be used as a substitute for professional medical advice.
            </div>
        </div>
        
        <footer>
            <p>COPD Detection System | Machine Learning for Healthcare</p>
        </footer>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file was uploaded
    if 'file' not in request.files:
        return '<div class="result" style="background-color: #f2dede; color: #a94442; padding: 20px; border-radius: 8px; margin: 20px; text-align: center;"><h2>Error</h2><p>No file uploaded</p><a href="/" style="display: inline-block; background-color: #5bc0de; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; margin-top: 15px;">Try Again</a></div>'
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return '<div class="result" style="background-color: #f2dede; color: #a94442; padding: 20px; border-radius: 8px; margin: 20px; text-align: center;"><h2>Error</h2><p>No file selected</p><a href="/" style="display: inline-block; background-color: #5bc0de; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; margin-top: 15px;">Try Again</a></div>'
    
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
                probability = prediction * 100
                if prediction > 0.5:
                    result_class = "COPD Detected"
                    result_message = f"The model detected signs of COPD with {probability:.2f}% confidence."
                    result_style = "background: linear-gradient(to right, #ff7675, #ff6b81); color: white;"
                    result_icon = "🔴"
                else:
                    result_class = "Normal"
                    result_message = f"The model classified this as normal with {100-probability:.2f}% confidence."
                    result_style = "background: linear-gradient(to right, #55a3ff, #3498db); color: white;"
                    result_icon = "🟢"
                
                # Generate Grad-CAM visualization
                gradcam_filename = "gradcam_" + unique_filename
                gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                
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
                        # Generate heatmap
                        heatmap = grad_cam(model, original_img, last_conv_layer_name)
                        # Save Grad-CAM visualization
                        visualize_grad_cam((original_img * 255).astype(np.uint8), heatmap, gradcam_filepath)
                        gradcam_available = True
                    else:
                        gradcam_available = False
                except Exception as e:
                    print(f"Grad-CAM error: {str(e)}")
                    gradcam_available = False
                
                # Create result HTML
                result_html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Prediction Result</title>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }}
                        .container {{ max-width: 900px; margin: 0 auto; padding: 30px; }}
                        header {{ text-align: center; padding: 30px 0; }}
                        h1 {{ color: #2c3e50; margin-bottom: 10px; font-size: 2.5em; }}
                        .card {{ background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 30px; }}
                        .result-box {{ {result_style} padding: 30px; border-radius: 15px; text-align: center; margin: 30px 0; box-shadow: 0 5px 20px rgba(0,0,0,0.15); }}
                        .result-box h2 {{ margin-top: 0; font-size: 2em; }}
                        .confidence {{ font-size: 1.2em; margin: 20px 0; }}
                        .raw-score {{ background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; display: inline-block; margin: 15px 0; }}
                        .images-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 30px 0; }}
                        .image-box {{ flex: 1; min-width: 300px; text-align: center; }}
                        .image-box h3 {{ color: #2c3e50; margin-bottom: 15px; }}
                        .image-box img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                        .btn {{ display: inline-block; background: linear-gradient(to right, #3498db, #2980b9); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 10px; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3); }}
                        .btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4); }}
                        .btn-secondary {{ background: linear-gradient(to right, #95a5a6, #7f8c8d); }}
                        .disclaimer {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; border-radius: 0 8px 8px 0; margin-top: 30px; font-size: 14px; }}
                        .disclaimer strong {{ color: #856404; }}
                        footer {{ text-align: center; padding: 20px; color: #7f8c8d; font-size: 0.9em; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <header>
                            <h1>Prediction Result</h1>
                        </header>
                        
                        <div class="card">
                            <div class="result-box">
                                <h2>{result_icon} {result_class}</h2>
                                <div class="confidence">{result_message}</div>
                                <div class="raw-score"><strong>Raw Score:</strong> {prediction:.4f} (0=Normal, 1=COPD)</div>
                            </div>
                            
                            <h2>Analysis Visualization</h2>
                            <div class="images-container">
                                <div class="image-box">
                                    <h3>Original X-Ray</h3>
                                    <img src="/uploads/{unique_filename}" alt="Original X-Ray">
                                </div>
                                
                                {f'''<div class="image-box">
                                    <h3>Grad-CAM Highlighted Areas</h3>
                                    <img src="/uploads/{gradcam_filename}" alt="Grad-CAM Visualization">
                                    <p style="margin-top: 10px; color: #7f8c8d; font-size: 0.9em;">Red/yellow areas indicate regions the AI focused on for its decision</p>
                                </div>''' if gradcam_available else '''<div class="image-box">
                                    <h3>Grad-CAM Visualization</h3>
                                    <p>Unable to generate visualization</p>
                                </div>'''}
                            </div>
                            
                            <div style="text-align: center; margin: 30px 0;">
                                <a href="/" class="btn">Analyze Another Image</a>
                                <a href="/uploads/{unique_filename}" download class="btn btn-secondary">Download Original Image</a>
                            </div>
                        </div>
                        
                        <div class="disclaimer">
                            <strong>⚠️ Medical Disclaimer:</strong> This model is intended for early screening and research purposes only and not for clinical diagnosis. Always consult with qualified healthcare professionals for medical concerns. The accuracy of this model has limitations and should not be used as a substitute for professional medical advice.
                        </div>
                    </div>
                    
                    <footer>
                        <p>COPD Detection System | Machine Learning for Healthcare</p>
                    </footer>
                </body>
                </html>
                '''
                
                return result_html
            else:
                return '<div class="result" style="background-color: #f2dede; color: #a94442; padding: 20px; border-radius: 8px; margin: 20px; text-align: center;"><h2>Error</h2><p>Model not loaded. Please check if the model file exists.</p><a href="/" style="display: inline-block; background-color: #5bc0de; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; margin-top: 15px;">Try Again</a></div>'
                
        except Exception as e:
            return f'<div class="result" style="background-color: #f2dede; color: #a94442; padding: 20px; border-radius: 8px; margin: 20px; text-align: center;"><h2>Error</h2><p>Error processing image: {str(e)}</p><a href="/" style="display: inline-block; background-color: #5bc0de; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; margin-top: 15px;">Try Again</a></div>'
    else:
        return '<div class="result" style="background-color: #f2dede; color: #a94442; padding: 20px; border-radius: 8px; margin: 20px; text-align: center;"><h2>Error</h2><p>Invalid file type. Please upload a JPG, JPEG, or PNG image.</p><a href="/" style="display: inline-block; background-color: #5bc0de; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; margin-top: 15px;">Try Again</a></div>'

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Import send_from_directory here to avoid circular import
    from flask import send_from_directory
    
    # Load model when app starts
    if load_model():
        # Run the app
        print("Starting Improved COPD Detection Web App...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(host='localhost', port=5000, debug=True)
    else:
        print("Failed to load model. Please check if 'best_copd_model.h5' exists in the current directory.")