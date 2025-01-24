import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


class ChestXray14Dataset(Dataset):
    def __init__(self, root_dir, image_list_file, transform=None):
        self.root_dir = root_dir
        self.image_labels = []
        self.transform = transform

        with open(image_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                self.image_labels.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label