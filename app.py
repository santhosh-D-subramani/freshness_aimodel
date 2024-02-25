from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)
model = tf.saved_model.load("./converted_savedmodel/model.savedmodel")

# Define the target size for images (adjust based on your model's input size)
TARGET_SIZE = (224, 224)

def preprocess_image(image_data):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))  # Resize to match the model's input shape
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict_mask(image_data):
    image_array = preprocess_image(image_data)

    # Make predictions using the loaded TensorFlow model with the specified signature
    predictions = model.signatures['serving_default'](tf.constant(image_array, dtype=tf.float32))

    # Assuming your model outputs a probability for mask presence (adjust based on your model)
    probability_mask = predictions['sequential_3'].numpy()[0][0]

    # You can define your own threshold for mask detection
    mask_detected = probability_mask > 0.5

    return {
        'Freshness': int(mask_detected),  # Convert boolean to integer
        'Freshness_probability': float(probability_mask)
    }

@app.route('/predict_mask', methods=['POST'])
def predict_mask_route():
    print("Request received")

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided in the request'}), 400

        base64_image = data['image']
        image_data = base64.b64decode(base64_image)
        
        print(f"Received image data size in predict_mask: {len(image_data)} bytes")
        result = predict_mask(image_data)
        print(f"Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
