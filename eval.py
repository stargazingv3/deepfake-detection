import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from timm import create_model
import argparse

# Define the image transformations for inference
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pre-trained model
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

def load_model(model_path):
    model = DeepFakeModel()
    print(model_path)
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def classify_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    
    probability = torch.sigmoid(output).item()  # Get the probability
    prediction = "fake" if probability > 0.5 else "real"
    confidence = probability if prediction == "fake" else 1 - probability  # Calculate confidence

    return prediction, confidence

def classify_folder(model, folder_path, device):
    i = 0
    results = {}
    for img_file in os.listdir(folder_path):
        i += 1
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_file)
            prediction, confidence = classify_image(model, img_path, device)
            results[img_file] = (prediction, confidence)
        if i > 100:
            break
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate images for deepfake classification")
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--files', type=str, nargs='+', help='Paths to specific image files')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model).to(device)

    if args.folder:
        results = classify_folder(model, args.folder, device)
        for img, (label, confidence) in results.items():
            print(f"{img}: {label} (Confidence: {confidence:.2f})")

    if args.files:
        for file in args.files:
            if os.path.isfile(file):
                label, confidence = classify_image(model, file, device)
                print(f"{os.path.basename(file)}: {label} (Confidence: {confidence:.2f})")
            else:
                print(f"File not found: {file}")

if __name__ == "__main__":
    main()
