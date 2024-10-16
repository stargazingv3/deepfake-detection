import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm import create_model

# Define a custom dataset class for loading images from directories
class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images and their labels from the directory structure
        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(root_dir, folder)
            for img_file in tqdm(os.listdir(folder_path), desc=f"Loading {folder} images", leave=False):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust as needed
                    self.images.append(os.path.join(folder_path, img_file))
                    self.labels.append(label)  # 0 for real, 1 for fake

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets for training, validation, and testing
train_dataset = DeepFakeDataset(root_dir='/data/elsa-10000/train', transform=transform)
val_dataset = DeepFakeDataset(root_dir='/data/elsa-10000/test', transform=transform)
test_dataset = DeepFakeDataset(root_dir='/data/elsa-10000/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Load a pre-trained Vision Transformer model
class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=True)  # Load a ViT model
        self.model.head = nn.Linear(self.model.head.in_features, 1)  # Adjust for binary classification

    def forward(self, x):
        return torch.sigmoid(self.model(x))  # Use sigmoid for binary output

# Set up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTModel().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # Wrap model for multi-GPU

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Update training and validation functions
def train(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)  # Move images to device
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images)  # Forward pass
        loss = criterion(outputs.view(-1), labels)  # Ensure outputs are reshaped
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = (outputs.view(-1) > 0.5).float()  # Reshape for predictions
        correct += (predictions.squeeze() == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)  # Move images to device
            labels = labels.float().to(device)
            outputs = model(images)  # Forward pass
            loss = criterion(outputs.view(-1), labels)  # Ensure outputs are reshaped
            total_loss += loss.item()
            predictions = (outputs.view(-1) > 0.5).float()  # Reshape for predictions
            correct += (predictions.squeeze() == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def test(model, dataloader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(device)  # Move images to device
            labels = labels.float().to(device)
            outputs = model(images)  # Forward pass
            predictions = (outputs > 0.5).float()
            correct += (predictions.squeeze() == labels).sum().item()

    return correct / len(dataloader.dataset)

# Training loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    train_loss, train_acc = train(model, criterion, optimizer, train_loader, device)
    val_loss, val_acc = validate(model, criterion, val_loader, device)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Testing the model
test_acc = test(model, test_loader, device)
print(f'Test Accuracy: {test_acc:.4f}')

# Save the trained model
torch.save(model.module.state_dict(), "deepfake_model.pth")
print("Model saved to deepfake_model.pth")
