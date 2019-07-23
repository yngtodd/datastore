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
