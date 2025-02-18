from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# Initialize Flask app (Move this line to the top before using `@app.route`)
app = Flask(__name__)

# Load the trained 3-class severity model
MODEL_PATH = "wheat_rust_model_severity.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 128
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')

# Upload & Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process image for prediction
    img = preprocess_image(file_path)
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)

    labels = ["Healthy", "Mild Rust", "Severe Rust"]
    treatments = [
        "No action needed. Keep monitoring.",
        "Apply fungicide and monitor for progression.",
        "Consider removing infected plants and using resistant wheat varieties."
    ]

    result = labels[class_idx]
    suggestion = treatments[class_idx]

    return jsonify({
        "result": result,
        "confidence": confidence,
        "suggestion": suggestion,
        "file_path": file_path
    })

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
