import os
import sys

import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, file_paths=None, transform=None):
        self.train_data = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.train_data.targets)

    def __getitem__(self, idx):
        image_paths, label = self.train_data.samples[idx]
        image = Image.open(image_paths)

        if self.transform:
            image = self.transform(image)

        return image, label
