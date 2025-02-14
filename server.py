from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import torch.nn as nn

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_path = 'MODEL_PATH_HERE'
model = load_model(model_path)

transform = transforms.Compose([
    transforms.Resize((224, 126)),
    transforms.Pad((49, 0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_single_image(model, image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        predicted_class = 1 if probability > 0.5 else 0
    return predicted_class, probability

@app.route('/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        filename = secure_filename(file.filename)
        image_bytes = file.read()
        predicted_class, probability = predict_single_image(model, image_bytes)
        return jsonify({
            'message': 'File processed successfully',
            'filename': filename,
            'predicted_class': predicted_class,
            'probability': probability
        }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)
