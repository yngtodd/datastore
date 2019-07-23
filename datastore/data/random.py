import numpy as np

from datastore.api import Dataset


class RandomData(Dataset):
    """ Random dataset - Useful for quick iterating """
    
    def __init__(self, num_samples: int, num_classes: int):
        self.data = np.random.randn(num_samples)
        self.label = np.random.randint(num_classes, size=num_samples)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]