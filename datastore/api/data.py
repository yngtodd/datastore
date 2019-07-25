from abc import abstractmethod


class Dataset:
    """ Abstract dataset - Used for both Keras and Pytorch"""

    @abstractmethod
    def __getitem__(self, idx):
        """Gets batch at position `index`.

        Parameters
        ----------
            idx: index position of the batch in the data.

        Returns
        -------
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Length of the dataset.

        Returns
        -------
            The number of samples in the data.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """ Keras method called at the end of every epoch. """
        pass

    def __iter__(self):
        """Create a generator that iterates over the data."""
        for item in (self[i] for i in range(len(self))):
            yield item


class InMemoryDataset(Dataset):
    """ Abstract class for in memory data """

    def load_data(self):
        """ Load data and labels """
        raise NotImplementedError


class MultiTaskMeta(type):
    """ Metaclass for Multitask Datasets """

    def __init__(cls, name, bases, dct):
        cls.labels = {}


class MultiTaskDataset(Dataset, metaclass=MultiTaskMeta):
    """ Abstract class for multitask datasets """

    def get_labels(self):
        return self.labels.keys()


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Parameters
    ----------
        dataset (Dataset): The whole Dataset

        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
