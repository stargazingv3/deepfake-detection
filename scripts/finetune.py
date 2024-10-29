import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm import create_model
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(root_dir, folder)
            for img_file in tqdm(os.listdir(folder_path), desc=f"Loading {folder} images", leave=False):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(folder_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros((3, 384, 384)), self.labels[idx]

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

# Data augmentations
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_datasets(root_dir):
    # Load all images and labels first
    all_images = []
    all_labels = []

    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(root_dir, folder)
        for img_file in tqdm(os.listdir(folder_path), desc=f"Loading {folder} images", leave=False):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(folder_path, img_file))
                all_labels.append(label)

    # Split into training, validation, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    val_images, test_images, val_labels, test_labels = train_test_split(
        test_images, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )

    # Create datasets using the original DeepFakeDataset class
    train_dataset = DeepFakeDataset(root_dir=root_dir, transform=train_transform)
    val_dataset = DeepFakeDataset(root_dir=root_dir, transform=val_transform)
    test_dataset = DeepFakeDataset(root_dir=root_dir, transform=val_transform)

    # Update the datasets with the newly split images and labels
    train_dataset.images = train_images
    train_dataset.labels = train_labels
    val_dataset.images = val_images
    val_dataset.labels = val_labels
    test_dataset.images = test_images
    test_dataset.labels = test_labels

    return train_dataset, val_dataset, test_dataset

# Fine-tuning setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeModel().to(device)

# Load the model checkpoint directly as a state dict
model_path = 'deepfake_model_best.pth'
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.train()  # Set model to training mode

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Create datasets and data loaders
train_dataset, val_dataset, test_dataset = create_datasets('datasets/Celeb-DF-v2-images/')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

scaler = GradScaler()

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()

        with autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    return total_loss / len(train_loader), correct / total, all_labels, all_predictions

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return total_loss / len(dataloader), correct / total, all_labels, all_predictions

def plot_confusion_matrix(labels, predictions, title, filename):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(filename)  # Save the confusion matrix
    plt.close()  # Close the plot to avoid displaying it

# Fine-tuning loop
num_epochs = 20
print("Starting fine-tuning...")
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    
    train_loss, train_acc, train_labels, train_predictions = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc, val_labels, val_predictions = validate(model, val_loader, criterion, device)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Generate unique filenames for the confusion matrices
    train_filename = f"train_confusion_matrix_epoch_{epoch + 1}.png"
    val_filename = f"val_confusion_matrix_epoch_{epoch + 1}.png"

    # Plot confusion matrices
    plot_confusion_matrix(train_labels, train_predictions, "Training Confusion Matrix", train_filename)
    plot_confusion_matrix(val_labels, val_predictions, "Validation Confusion Matrix", val_filename)

# Test the model
model.eval()
test_labels = []
test_predictions = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images)
        predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()

        test_labels.extend(labels.cpu().numpy())
        test_predictions.extend(predictions.cpu().numpy())

# Generate a unique filename for the test confusion matrix
test_filename = "test_confusion_matrix.png"

# Plot test confusion matrix
plot_confusion_matrix(test_labels, test_predictions, "Test Confusion Matrix", test_filename)

# Save the fine-tuned model
torch.save(model.state_dict(), "finetuned.pth")
print("Fine-tuned model saved to finetuned.pth.")
