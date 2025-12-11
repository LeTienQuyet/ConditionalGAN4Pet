import os

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from PIL import Image

class PetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Dog = 0, Cat = 1
        self.class_to_idx = {"dog": 0, "cat": 1}

        for cls_name, label in self.class_to_idx.items():
            cls_folder = os.path.join(data_dir, cls_name)
            for file_name in os.listdir(cls_folder):
                file_path = os.path.join(cls_folder, file_name)
                self.image_files.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def prepare_data(root_dir="datasets", batch_size=128):
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_dev = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    train_dataset = PetDataset(os.path.join(root_dir, "train"), transform_train)
    dev_dataset = PetDataset(os.path.join(root_dir, "val"), transform_dev)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return train_dataloader, dev_dataloader