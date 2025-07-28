import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

class LeftCrop: 
    def __init__(self, width):
        self.width = width
    
    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, self.width, h))

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        for file in os.listdir(root_dir):
            if file.lower().endswith(valid_extensions):
                self.images.append(os.path.join(root_dir, file))

        print(f"find {len(self.images)} images，adjust to 256x256")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
