import os
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load disease information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')

# Load the trained model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))
model.eval()

def predict_disease(image_path):
    """Predict the disease from the given image."""
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
        output = model(input_data)
        _, predicted_idx = torch.max(output, 1)
        return predicted_idx.item()
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise e

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def submit():
    print("Request received!")
    
    if 'image' not in request.files:
        print("Error: No image provided")
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    if image.filename == '':
        print("Error: No selected file")
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded image to a temporary file
        temp_dir = 'static/uploads'
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, image.filename)
        image.save(file_path)
        print(f"Image saved at {file_path}")

        # Predict the disease
        pred_idx = predict_disease(file_path)
        disease_name = disease_info.iloc[pred_idx]['disease_name']
        print(f"Prediction: {disease_name}")
        
        return jsonify({'disease_name': disease_name})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)