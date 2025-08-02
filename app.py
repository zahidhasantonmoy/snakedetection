
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the models
try:
    efficientnet_model = load_model('final_efficientnetb0_classifier.h5')
    mobilenet_model = load_model('MobileNetV2_best_model.h5')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    efficientnet_model = None
    mobilenet_model = None

# Define image preprocessing function
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalize to [0,1]
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        results = {}

        if efficientnet_model:
            processed_image = preprocess_image(filepath, target_size=(224, 224))
            efficientnet_pred = efficientnet_model.predict(processed_image)
            # Assuming binary classification, adjust as needed for your model's output
            efficientnet_class = "Class 1" if efficientnet_pred[0][0] > 0.5 else "Class 0"
            efficientnet_confidence = float(efficientnet_pred[0][0])
            results['efficientnet'] = {'class': efficientnet_class, 'confidence': efficientnet_confidence}
        else:
            results['efficientnet'] = {'error': 'Model not loaded'}

        if mobilenet_model:
            processed_image = preprocess_image(filepath, target_size=(224, 224))
            mobilenet_pred = mobilenet_model.predict(processed_image)
            # Assuming binary classification, adjust as needed for your model's output
            mobilenet_class = "Class A" if mobilenet_pred[0][0] > 0.5 else "Class B"
            mobilenet_confidence = float(mobilenet_pred[0][0])
            results['mobilenet'] = {'class': mobilenet_class, 'confidence': mobilenet_confidence}
        else:
            results['mobilenet'] = {'error': 'Model not loaded'}

        # Clean up the uploaded file
        os.remove(filepath)

        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
