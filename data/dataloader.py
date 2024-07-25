"""DataLoader for LicensePlateDataset"""

from torch import cuda
from torch.utils.data import DataLoader

class LicensePlateDataLoader(DataLoader):
    def __init__(self, dataset, batch_size = 64, shuffle = True):
        pin_memory = False
        pin_memory_device = "cuda"

        if cuda.is_available():
            pin_memory = True

        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=pin_memory, pin_memory_device=pin_memory_device)
