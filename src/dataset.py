import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultiLabelViolenceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Колонки з мітками: NonViolence, Violence, guns, knife
        self.label_columns = ['NonViolence', 'Violence', 'guns', 'knife']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        # Беремо всі 4 мітки як список чисел
        labels = self.annotations.iloc[index, 1:5].values.astype('float32')
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels

def get_dataloaders(data_dir, batch_size=32, img_size=128):
    # Enhanced augmentation for training data
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Basic transforms for validation data (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    train_loader = DataLoader(
        MultiLabelViolenceDataset(os.path.join(data_dir, 'train', '_classes.csv'),
                                  os.path.join(data_dir, 'train'), train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
        prefetch_factor=2, persistent_workers=True
    )

    val_loader = DataLoader(
        MultiLabelViolenceDataset(os.path.join(data_dir, 'valid', '_classes.csv'),
                                  os.path.join(data_dir, 'valid'), val_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
        prefetch_factor=2, persistent_workers=True
    )

    return train_loader, val_loader