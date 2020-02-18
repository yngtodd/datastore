import torch
import numpy as np
import pandas as pd

from datastore.api import InMemoryDataset, MultiTaskDataset


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


class RandomMultiTaskData(InMemoryDataset, MultiTaskDataset):
    """ Random multiclass dataset - Useful for quick iterating """

    def __init__(self, num_samples: int, num_tasks: int, num_classes: int, seed: int=13):
        np.random.seed(seed)
        self.data = torch.randn([num_samples, 10])
        self._create_labels(num_tasks, num_classes, num_samples)
        
    def _create_labels(self, num_tasks, num_classes, num_samples):
        for i in range(num_tasks):
            self.labels[f'task{i}'] = np.random.randint(
                num_classes, 
                size=num_samples
            )

    def load_data(self):
        return self.data, self.labels

    def __repr__(self):
        return f'Random multitask supervised dataset'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.index_labels(idx)
