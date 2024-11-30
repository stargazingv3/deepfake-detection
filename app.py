import os
from flask import Flask, request, jsonify
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import io
import base64
from io import BytesIO

class VisionTransformerModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=1):
        super(VisionTransformerModel, self).__init__()
        self.model = create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.model.num_features
        
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.model(x)
        features = self.dropout(features)
        return self.head(features)

# Load model
def load_VIT_model(model_path):
    model = VisionTransformerModel()
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Extract the actual state dict from the "model" key
    model_state_dict = state_dict['model']
    
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

app = Flask(__name__)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    # Convert RGBA to RGB
    transforms.Lambda(lambda x: x[:3] if x.shape[0] == 4 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Model Loading
class DeepFakeModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_xl_in21ft1k'):
        super(DeepFakeModel, self).__init__()
        self.model = create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.model.num_features
        
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.model(x)
        features = self.dropout(features)
        return self.head(features)

# Load model
def load_model(model_path):
    model = DeepFakeModel()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model('deepfake_model_best.pth')  # Adjust this to your model path

# Helper to convert base64 image to PIL image
def decode_base64(base64_str):
    # Strip the data URI prefix if it exists (e.g. data:image/jpeg;base64,)
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
        
    img_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(img_data))
    
    # Convert image to RGB
    return image.convert('RGB')

@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.get_json()
    image_base64 = data['imageBase64']
    print(f"Received image data: {image_base64[:100]}...")  # Log first 100 chars of the Base64 string for debugging
    
    try:
        image = decode_base64(image_base64)
    except Exception as e:
        return jsonify({'error': f"Failed to decode image: {str(e)}"}), 400
    
    # Preprocess image
    image = transform(image).unsqueeze(0)
    
    # Model inference
    output = model(image)
    probability = torch.sigmoid(output).item()
    prediction = "fake" if probability > 0.05 else "real"
    confidence = probability if prediction == "fake" else 1 - probability
    print("Prediction: ", prediction)
    print("Probability: ", probability)
    print("Confidence: ", confidence)
    return jsonify({'prediction': prediction, 'probability': probability})


if __name__ == '__main__':
    # Bind Flask to all network interfaces (0.0.0.0) to allow external access
    app.run(host="0.0.0.0", port=5000)

