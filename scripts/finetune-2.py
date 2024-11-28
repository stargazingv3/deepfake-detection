import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm import create_model
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, folder_name, transform=None):
        self.root_dir = root_dir
        self.folder_name = folder_name
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Determine label based on the folder (real or fake)
        if 'fake' in folder_name:
            label = 1  # Fake images are labeled as 1
        else:
            label = 0  # Real images are labeled as 0

        folder_path = os.path.join(root_dir, folder_name)
        for img_file in tqdm(os.listdir(folder_path), desc=f"Loading {folder_name} images", leave=False):
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
    # Create datasets for the four folders: fake_train, real_train, fake_test, real_test
    train_dataset_fake = DeepFakeDataset(root_dir, 'fake_train', transform=train_transform)
    train_dataset_real = DeepFakeDataset(root_dir, 'real_train', transform=train_transform)
    val_dataset_fake = DeepFakeDataset(root_dir, 'fake_test', transform=val_transform)
    val_dataset_real = DeepFakeDataset(root_dir, 'real_test', transform=val_transform)

    # Combine real and fake datasets for train and validation
    train_dataset = train_dataset_fake + train_dataset_real
    val_dataset = val_dataset_fake + val_dataset_real

    return train_dataset, val_dataset

# Fine-tuning setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeModel().to(device)

# Load the model checkpoint directly as a state dict
model_path = 'finetuned-CELEBDF.pth'
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.train()  # Set model to training mode

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Create datasets and data loaders
root_dir = 'datasets/deepfake_in_the_wild'  # Path to your dataset folder
train_dataset, val_dataset = create_datasets(root_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

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

# Fine-tuning loop
num_epochs = 5
print("Starting fine-tuning...")
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    
    train_loss, train_acc, train_labels, train_predictions = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss, val_acc, val_labels, val_predictions = validate(model, val_loader, criterion, device)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    torch.save(model.state_dict(), "finetuned-wild.pth")
    
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

# Save the fine-tuned model
torch.save(model.state_dict(), "finetuned-wild.pth")
print("Fine-tuned model saved to finetuned.pth.")
