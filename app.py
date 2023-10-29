from flask import Flask, request, jsonify, send_file, Response, json
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = [
    'Bacterial leaf blight', 
    'Brown spot', 
    'Healthy', 
    'Leaf Blast', 
    'Leaf Scald', 'Narrow Brown Spot']

# Load the pre-trained ViT model
model = torch.load('model/vit.pth', map_location=torch.device('cpu'))

model.eval()





def predict_image(image):
    # Apply the same preprocessing transform as before
    transformed_image = transform(image)

    # Forward pass through the model
    with torch.no_grad():
        predictions = model(transformed_image.unsqueeze(0))
    
    probabilities = predictions.logits  # You may need to check the exact key used in your model's output
    predicted_class = probabilities.argmax(dim=-1).item()
    predicted_label = class_names[predicted_class]

    return predicted_label



app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'Uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/pred', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Ensure the filename is secure to prevent malicious uploads
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Read the image
        im = Image.open(filename)
        predictions = predict_image(im)

        response = Response(response=json.dumps({'predictions': predictions}), status=200)
        return response

@app.route('/')
def hello_world():
    response = Response(json.dumps({'message': 'Working Ok'}), status=200)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
