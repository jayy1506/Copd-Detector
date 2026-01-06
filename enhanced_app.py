from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Simple user database (in production, use a real database)
users = {
    'admin': 'password123',
    'user': 'mypassword'
}

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
    if 'username' in session:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Login - COPD Detection</title>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #1e3c72, #2a5298); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
                    .login-container { background: white; border-radius: 15px; box-shadow: 0 15px 30px rgba(0,0,0,0.2); padding: 40px; width: 400px; }
                    h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
                    .form-group { margin-bottom: 20px; }
                    label { display: block; margin-bottom: 8px; font-weight: 600; color: #34495e; }
                    input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; box-sizing: border-box; transition: border-color 0.3s; }
                    input:focus { border-color: #3498db; outline: none; }
                    button { width: 100%; padding: 14px; background: linear-gradient(to right, #3498db, #2980b9); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
                    button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4); }
                    .error { color: #e74c3c; text-align: center; margin-bottom: 20px; }
                    .signup-link { text-align: center; margin-top: 20px; color: #7f8c8d; }
                    .signup-link a { color: #3498db; text-decoration: none; }
                    .signup-link a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="login-container">
                    <h1>🫁 COPD Detection Login</h1>
                    <div class="error">Invalid username or password</div>
                    <form method="POST">
                        <div class="form-group">
                            <label for="username">Username</label>
                            <input type="text" id="username" name="username" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password</label>
                            <input type="password" id="password" name="password" required>
                        </div>
                        <button type="submit">Login</button>
                    </form>
                    <div class="signup-link">
                        Don't have an account? <a href="/signup">Sign Up</a>
                    </div>
                </div>
            </body>
            </html>
            '''
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login - COPD Detection</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #1e3c72, #2a5298); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .login-container { background: white; border-radius: 15px; box-shadow: 0 15px 30px rgba(0,0,0,0.2); padding: 40px; width: 400px; }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #34495e; }
            input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; box-sizing: border-box; transition: border-color 0.3s; }
            input:focus { border-color: #3498db; outline: none; }
            button { width: 100%; padding: 14px; background: linear-gradient(to right, #3498db, #2980b9); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4); }
            .signup-link { text-align: center; margin-top: 20px; color: #7f8c8d; }
            .signup-link a { color: #3498db; text-decoration: none; }
            .signup-link a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h1>🫁 COPD Detection Login</h1>
            <form method="POST">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit">Login</button>
            </form>
            <div class="signup-link">
                Don't have an account? <a href="/signup">Sign Up</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sign Up - COPD Detection</title>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #1e3c72, #2a5298); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
                    .signup-container { background: white; border-radius: 15px; box-shadow: 0 15px 30px rgba(0,0,0,0.2); padding: 40px; width: 400px; }
                    h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
                    .form-group { margin-bottom: 20px; }
                    label { display: block; margin-bottom: 8px; font-weight: 600; color: #34495e; }
                    input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; box-sizing: border-box; transition: border-color 0.3s; }
                    input:focus { border-color: #3498db; outline: none; }
                    button { width: 100%; padding: 14px; background: linear-gradient(to right, #27ae60, #219653); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
                    button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4); }
                    .error { color: #e74c3c; text-align: center; margin-bottom: 20px; }
                    .login-link { text-align: center; margin-top: 20px; color: #7f8c8d; }
                    .login-link a { color: #3498db; text-decoration: none; }
                    .login-link a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="signup-container">
                    <h1>🫁 COPD Detection Sign Up</h1>
                    <div class="error">Passwords do not match</div>
                    <form method="POST">
                        <div class="form-group">
                            <label for="username">Username</label>
                            <input type="text" id="username" name="username" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password</label>
                            <input type="password" id="password" name="password" required>
                        </div>
                        <div class="form-group">
                            <label for="confirm_password">Confirm Password</label>
                            <input type="password" id="confirm_password" name="confirm_password" required>
                        </div>
                        <button type="submit">Sign Up</button>
                    </form>
                    <div class="login-link">
                        Already have an account? <a href="/login">Login</a>
                    </div>
                </div>
            </body>
            </html>
            '''
        
        if username in users:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sign Up - COPD Detection</title>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #1e3c72, #2a5298); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
                    .signup-container { background: white; border-radius: 15px; box-shadow: 0 15px 30px rgba(0,0,0,0.2); padding: 40px; width: 400px; }
                    h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
                    .form-group { margin-bottom: 20px; }
                    label { display: block; margin-bottom: 8px; font-weight: 600; color: #34495e; }
                    input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; box-sizing: border-box; transition: border-color 0.3s; }
                    input:focus { border-color: #3498db; outline: none; }
                    button { width: 100%; padding: 14px; background: linear-gradient(to right, #27ae60, #219653); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
                    button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4); }
                    .error { color: #e74c3c; text-align: center; margin-bottom: 20px; }
                    .login-link { text-align: center; margin-top: 20px; color: #7f8c8d; }
                    .login-link a { color: #3498db; text-decoration: none; }
                    .login-link a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <div class="signup-container">
                    <h1>🫁 COPD Detection Sign Up</h1>
                    <div class="error">Username already exists</div>
                    <form method="POST">
                        <div class="form-group">
                            <label for="username">Username</label>
                            <input type="text" id="username" name="username" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password</label>
                            <input type="password" id="password" name="password" required>
                        </div>
                        <div class="form-group">
                            <label for="confirm_password">Confirm Password</label>
                            <input type="password" id="confirm_password" name="confirm_password" required>
                        </div>
                        <button type="submit">Sign Up</button>
                    </form>
                    <div class="login-link">
                        Already have an account? <a href="/login">Login</a>
                    </div>
                </div>
            </body>
            </html>
            '''
        
        # Add new user
        users[username] = password
        session['username'] = username
        return redirect(url_for('upload'))
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sign Up - COPD Detection</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #1e3c72, #2a5298); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .signup-container { background: white; border-radius: 15px; box-shadow: 0 15px 30px rgba(0,0,0,0.2); padding: 40px; width: 400px; }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #34495e; }
            input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; box-sizing: border-box; transition: border-color 0.3s; }
            input:focus { border-color: #3498db; outline: none; }
            button { width: 100%; padding: 14px; background: linear-gradient(to right, #27ae60, #219653); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4); }
            .login-link { text-align: center; margin-top: 20px; color: #7f8c8d; }
            .login-link a { color: #3498db; text-decoration: none; }
            .login-link a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="signup-container">
            <h1>🫁 COPD Detection Sign Up</h1>
            <form method="POST">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <div class="form-group">
                    <label for="confirm_password">Confirm Password</label>
                    <input type="password" id="confirm_password" name="confirm_password" required>
                </div>
                <button type="submit">Sign Up</button>
            </form>
            <div class="login-link">
                Already have an account? <a href="/login">Login</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/upload')
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload - COPD Detection</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }
            .container { max-width: 1200px; margin: 0 auto; padding: 30px; }
            header { display: flex; justify-content: space-between; align-items: center; padding: 20px 0; }
            h1 { color: #2c3e50; margin: 0; }
            .logout-btn { background: #e74c3c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
            .logout-btn:hover { background: #c0392b; }
            .main-content { display: flex; gap: 30px; margin-top: 30px; }
            .upload-section { flex: 1; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 30px; }
            .result-section { flex: 1; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 30px; min-height: 400px; }
            .upload-area { border: 3px dashed #3498db; border-radius: 10px; padding: 40px; text-align: center; margin: 30px 0; transition: all 0.3s ease; background-color: #f8f9fa; }
            .upload-area:hover { border-color: #2980b9; background-color: #e3f2fd; }
            .upload-area h2 { color: #3498db; margin-top: 0; }
            input[type="file"] { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; width: 100%; box-sizing: border-box; }
            input[type="submit"] { background: linear-gradient(to right, #3498db, #2980b9); color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3); }
            input[type="submit"]:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4); }
            .disclaimer { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; border-radius: 0 8px 8px 0; margin-top: 30px; font-size: 14px; }
            .disclaimer strong { color: #856404; }
            .image-preview { max-width: 100%; max-height: 300px; margin: 20px 0; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .placeholder { color: #7f8c8d; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🫁 COPD Detection System</h1>
                <a href="/logout" class="logout-btn">Logout</a>
            </header>
            
            <div class="main-content">
                <div class="upload-section">
                    <h2>Upload Chest X-Ray Image</h2>
                    <p>Select a chest X-ray image for COPD analysis</p>
                    
                    <form method="post" action="/predict" enctype="multipart/form-data">
                        <div class="upload-area">
                            <h2>📁 Drop your X-Ray image here</h2>
                            <p>or</p>
                            <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
                        </div>
                        <input type="submit" value="Analyze X-Ray">
                    </form>
                </div>
                
                <div class="result-section">
                    <h2>Analysis Results</h2>
                    <div class="placeholder">
                        <p>Upload an image to see the analysis results and Grad-CAM visualization.</p>
                        <p>The results will show:</p>
                        <ul>
                            <li>Prediction (Normal or COPD)</li>
                            <li>Confidence score</li>
                            <li>Original image</li>
                            <li>Grad-CAM highlighted areas</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="disclaimer">
                <strong>⚠️ Medical Disclaimer:</strong> This model is intended for early screening and research purposes only and not for clinical diagnosis. Always consult with qualified healthcare professionals for medical concerns. The accuracy of this model has limitations and should not be used as a substitute for professional medical advice.
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    # Check if file was uploaded
    if 'file' not in request.files:
        return '''
        <script>
            alert("No file uploaded");
            window.location.href = "/upload";
        </script>
        '''
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return '''
        <script>
            alert("No file selected");
            window.location.href = "/upload";
        </script>
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
                    <title>Results - COPD Detection</title>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }}
                        .container {{ max-width: 1200px; margin: 0 auto; padding: 30px; }}
                        header {{ display: flex; justify-content: space-between; align-items: center; padding: 20px 0; }}
                        h1 {{ color: #2c3e50; margin: 0; }}
                        .logout-btn {{ background: #e74c3c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
                        .logout-btn:hover {{ background: #c0392b; }}
                        .back-btn {{ background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
                        .back-btn:hover {{ background: #2980b9; }}
                        .nav-buttons {{ display: flex; gap: 10px; }}
                        .main-content {{ display: flex; gap: 30px; margin-top: 30px; }}
                        .image-section {{ flex: 1; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 30px; }}
                        .result-section {{ flex: 1; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 30px; }}
                        .result-box {{ {result_style} padding: 30px; border-radius: 15px; text-align: center; margin: 30px 0; box-shadow: 0 5px 20px rgba(0,0,0,0.15); }}
                        .result-box h2 {{ margin-top: 0; font-size: 2em; }}
                        .confidence {{ font-size: 1.2em; margin: 20px 0; }}
                        .raw-score {{ background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; display: inline-block; margin: 15px 0; }}
                        .image-container {{ text-align: center; margin: 20px 0; }}
                        .image-container img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                        .image-label {{ font-weight: bold; margin: 10px 0; color: #2c3e50; }}
                        .disclaimer {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; border-radius: 0 8px 8px 0; margin-top: 30px; font-size: 14px; }}
                        .disclaimer strong {{ color: #856404; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <header>
                            <h1>🫁 COPD Detection Results</h1>
                            <div class="nav-buttons">
                                <a href="/upload" class="back-btn">← Back to Upload</a>
                                <a href="/logout" class="logout-btn">Logout</a>
                            </div>
                        </header>
                        
                        <div class="main-content">
                            <div class="image-section">
                                <h2>Image Analysis</h2>
                                <div class="image-container">
                                    <div class="image-label">Original X-Ray Image</div>
                                    <img src="/{filepath}" alt="Original X-Ray">
                                </div>
                                
                                <div class="image-container">
                                    <div class="image-label">Grad-CAM Highlighted Areas</div>
                                    {f'<img src="/{gradcam_filepath}" alt="Grad-CAM Visualization">' if gradcam_available else '<p>Unable to generate visualization</p>'}
                                    <p style="margin-top: 10px; color: #7f8c8d; font-size: 0.9em;">Red/yellow areas indicate regions the AI focused on for its decision</p>
                                </div>
                            </div>
                            
                            <div class="result-section">
                                <h2>Prediction Results</h2>
                                <div class="result-box">
                                    <h2>{result_icon} {result_class}</h2>
                                    <div class="confidence">{result_message}</div>
                                    <div class="raw-score"><strong>Raw Score:</strong> {prediction:.4f} (0=Normal, 1=COPD)</div>
                                </div>
                                
                                <h3>Interpretation Guide</h3>
                                <ul>
                                    <li><strong>Normal (🟢):</strong> No signs of COPD detected</li>
                                    <li><strong>COPD Detected (🔴):</strong> Signs of chronic obstructive pulmonary disease</li>
                                    <li><strong>Confidence Score:</strong> How certain the AI is in its prediction</li>
                                    <li><strong>Grad-CAM Areas:</strong> Regions that influenced the AI's decision</li>
                                </ul>
                                
                                <div style="margin-top: 30px; text-align: center;">
                                    <a href="/upload" class="back-btn" style="padding: 15px 30px; font-size: 18px;">Analyze Another Image</a>
                                </div>
                            </div>
                        </div>
                        
                        <div class="disclaimer">
                            <strong>⚠️ Medical Disclaimer:</strong> This model is intended for early screening and research purposes only and not for clinical diagnosis. Always consult with qualified healthcare professionals for medical concerns. The accuracy of this model has limitations and should not be used as a substitute for professional medical advice.
                        </div>
                    </div>
                </body>
                </html>
                '''
                
                return result_html
            else:
                return '''
                <script>
                    alert("Model not loaded. Please check if the model file exists.");
                    window.location.href = "/upload";
                </script>
                '''
                
        except Exception as e:
            return f'''
            <script>
                alert("Error processing image: {str(e)}");
                window.location.href = "/upload";
            </script>
            '''
    else:
        return '''
        <script>
            alert("Invalid file type. Please upload a JPG, JPEG, or PNG image.");
            window.location.href = "/upload";
        </script>
        '''

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Import send_from_directory here to avoid circular import
    from flask import send_from_directory
    
    # Load model when app starts
    if load_model():
        # Run the app
        print("Starting Enhanced COPD Detection Web App...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(host='localhost', port=5000, debug=True)
    else:
        print("Failed to load model. Please check if 'best_copd_model.h5' exists in the current directory.")