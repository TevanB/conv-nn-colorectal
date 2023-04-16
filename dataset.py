import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, RandomRotation

class ColorectalCancerDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Convert the NumPy array to a PIL Image
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.img_paths)
