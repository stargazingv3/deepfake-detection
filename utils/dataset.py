# utils/dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(folder_path, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
