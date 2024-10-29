import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm import create_model

# Define a custom dataset class for loading images
class DeepFakeDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Load the model
class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTModel().to(device)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset and split into train, val, and test sets
def load_dataset(base_dir, num_samples=1000):
    images = []
    labels = []

    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(base_dir, folder)
        for img_file in tqdm(os.listdir(folder_path), desc=f"Loading {folder} images", leave=False):
            if img_file.endswith(('.png', '.jpg', '.jpeg')) and len(images) < num_samples:
                images.append(os.path.join(folder_path, img_file))
                labels.append(label)

    return images, labels

# Load dataset from the specified directory
base_dir = 'datasets/whichfaceisreal'
images, labels = load_dataset(base_dir)

# Split dataset into train, val, and test
random.seed(42)  # For reproducibility
indices = list(range(len(images)))
random.shuffle(indices)

train_size = int(0.7 * len(images))
val_size = int(0.15 * len(images))
test_size = len(images) - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_images = [images[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]
val_images = [images[i] for i in val_indices]
val_labels = [labels[i] for i in val_indices]
test_images = [images[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

# Create DataLoaders
train_dataset = DeepFakeDataset(train_images, train_labels, transform=transform)
val_dataset = DeepFakeDataset(val_images, val_labels, transform=transform)
test_dataset = DeepFakeDataset(test_images, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set up training
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (outputs.view(-1) > 0.5).float()
        correct += (predictions.squeeze() == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            total_loss += loss.item()
            predictions = (outputs.view(-1) > 0.5).float()
            correct += (predictions.squeeze() == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# Test function
def test(model, dataloader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            predictions = (outputs > 0.5).float()
            correct += (predictions.squeeze() == labels).sum().item()

    return correct / len(dataloader.dataset)

# Training loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Testing the model
test_acc = test(model, test_loader, device)
print(f'Test Accuracy: {test_acc:.4f}')

# Save the model weights after training and testing
torch.save(model.state_dict(), "deepfake_model_trained_and_tested.pth")
print("Model weights saved to deepfake_model_trained_and_tested.pth")
