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
from torch.nn import functional as F

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
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
            # First load the image
            image = Image.open(img_path)
            
            # Handle transparency
            if image.mode == 'RGBA':
                # Convert RGBA to RGB with white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
                image = background
            elif image.mode == 'P':  # Palette mode
                # Convert P to RGBA first if there's transparency
                image = image.convert('RGBA')
                # Then convert to RGB with white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != 'RGB':
                # Convert any other mode to RGB
                image = image.convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default tensor if image loading fails
            return torch.zeros((3, 384, 384)), self.labels[idx]

class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = len(images)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        label_a, label_b = labels, labels[index]
        return mixed_images, label_a, label_b, lam

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
        # Removed sigmoid activation since we're using BCEWithLogitsLoss
        return self.head(features)

# Advanced augmentations
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(15),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# Create datasets
train_dataset = DeepFakeDataset(root_dir='/data/elsa-100000/train', transform=train_transform, train=True)
val_dataset = DeepFakeDataset(root_dir='/data/elsa-100000/val', transform=val_transform)
test_dataset = DeepFakeDataset(root_dir='/data/elsa-100000/test', transform=val_transform)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=8, 
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=8, 
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=8, 
    pin_memory=True
)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeModel().to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Changed to BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos'
)

mixup = Mixup(alpha=0.2)
scaler = GradScaler()

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.float().to(device)
        mixed_images, labels_a, labels_b, lam = mixup((images, labels))
        
        optimizer.zero_grad()
        
        with autocast(enabled=True):
            outputs = model(mixed_images)
            loss = lam * criterion(outputs.view(-1), labels_a) + (1 - lam) * criterion(outputs.view(-1), labels_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        # Apply sigmoid here for accuracy calculation
        predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
        correct += (lam * (predictions == labels_a).float() + (1 - lam) * (predictions == labels_b).float()).sum().item()
        total += len(labels)

    return total_loss / len(train_loader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            
            total_loss += loss.item()
            # Apply sigmoid here for accuracy calculation
            predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return total_loss / len(dataloader), correct / total

# Training loop with early stopping
num_epochs = 20
best_val_acc = 0
patience = 5
patience_counter = 0

print("Starting training...")
print(f"Training on device: {device}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        '''torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, "deepfake_model_best.pth")'''
        torch.save(model.module.state_dict(), "deepfake_model_best.pth")
        print("Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Final evaluation
print("Running final evaluation...")
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f'Final Test Accuracy: {test_acc:.4f}')