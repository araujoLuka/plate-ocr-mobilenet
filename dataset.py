"""Dataset class for License Plate Recognition project."""

import os
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LicensePlateDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = os.listdir(image_folder)
        self.classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.images[idx].split('_')[0]  # Assumindo que o nome do arquivo é "label_xxx.jpg"
        label_idx = self.class_to_idx[label]  # Convertendo label para índice numérico

        if self.transform:
            image = self.transform(image)

        return image, tensor(label_idx)  # Convertendo label para tensor

    def genDataLoader(self, batch_size, shuffle) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
