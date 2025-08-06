from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from snake_info import snake_data

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

# Get class names from snake_data
class_names = list(snake_data.keys())

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
    model_name = request.form.get('model', 'efficientnet')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        results = {}
        model = None
        if model_name == 'efficientnet':
            model = efficientnet_model
        elif model_name == 'mobilenet':
            model = mobilenet_model

        if model:
            try:
                processed_image = preprocess_image(filepath, target_size=(224, 224))
                predictions = model.predict(processed_image)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_class_name = class_names[predicted_class_index]
                confidence = float(predictions[0][predicted_class_index])
                snake_info = snake_data[predicted_class_name]
                results[model_name] = {
                    'class': predicted_class_name,
                    'confidence': confidence,
                    'venomous': snake_info['venomous'],
                    'risk_level': snake_info['risk_level'],
                    'first_aid': snake_info['first_aid']
                }
            except Exception as e:
                results[model_name] = {'error': f'Prediction error: {e}'}
        else:
            results[model_name] = {'error': 'Model not loaded'}

        # Clean up the uploaded file
        os.remove(filepath)

        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
