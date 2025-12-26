import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_data_loaders(data_dir, batch_size=32, val_split=0.2):

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
       
        train_dir = data_dir
        for root, dirs, files in os.walk(data_dir):
            if "Train" in dirs:
                train_dir = os.path.join(root, "Train")
                print(f"Detected Train directory at: {train_dir}")
                break
      
        full_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
        
       
        print(f"Classes found: {full_dataset.classes}")
       
        dataset_size = len(full_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, full_dataset.classes

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
