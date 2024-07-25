"""Dataset class for License Plate Recognition project."""

import os
from torch import cuda, tensor
from torch.utils.data import Dataset
from PIL import Image
from assets.metadata import CLASS_TO_IDX, CATEGORIES
from utils.transforms import baseTransform 

class OCRDataset(Dataset):
    def __init__(self, image_folder, transform=baseTransform):
        self.image_folder = image_folder
        self.transform = transform
        self.images = self.__listimages(image_folder)
        self.classes = CATEGORIES
        self.class_to_idx = CLASS_TO_IDX

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.images[idx].split('/')[0]  # Filename is "<l>/xxx.png" where l is the char
        label_idx = self.class_to_idx[label]
        image = self.transform(image)
        tensor_label = tensor(label_idx)
        return image, tensor_label

    def __listimages(self, image_folder: str) -> list[str]:
        # Training images are in subfolders named after the class
        classDirs = [d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]
        images = []
        for d in classDirs:
            for f in os.listdir(os.path.join(image_folder, d)):
                if f.endswith('.png'):
                    images.append(f'{d}/{f}')
        return images
