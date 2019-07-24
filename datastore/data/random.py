import numpy as np

from datastore.api import InMemoryDataset


class RandomData(InMemoryDataset):
    """ Random dataset - Useful for quick iterating """

    def __init__(self, num_samples: int, num_classes: int, seed: int=13):
        np.random.seed(seed)
        self.data = np.random.randn(num_samples)
        self.labels = np.random.randint(num_classes, size=num_samples)

    def load_data(self):
        return self.data, self.labels

    def __repr__(self):
        return f'Random supervised dataset'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
