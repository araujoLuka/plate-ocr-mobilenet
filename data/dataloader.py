"""DataLoader for LicensePlateDataset"""

from torch.utils.data import DataLoader

class LicensePlateDataLoader(DataLoader):
    def __init__(self, dataset, batch_size = 32, shuffle = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
