import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from timm import create_model
import argparse

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pre-trained model
class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

def load_model(model_path):
    model = ViTModel()
    # Load the state dictionary with weights_only=True
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def classify_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    prediction = (output > 0.5).item()
    return "fake" if prediction else "real"

def classify_folder(model, folder_path, device):
    results = {}
    for img_file in os.listdir(folder_path):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_file)
            result = classify_image(model, img_path, device)
            results[img_file] = result
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
        for img, label in results.items():
            print(f"{img}: {label}")

    if args.files:
        for file in args.files:
            if os.path.isfile(file):
                label = classify_image(model, file, device)
                print(f"{os.path.basename(file)}: {label}")
            else:
                print(f"File not found: {file}")

if __name__ == "__main__":
    main()
