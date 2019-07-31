import math
import numpy as np

from collections import namedtuple
from sklearn.utils import resample

from datastore.api.data import Subset


def dummy_indices(dataset):
    """ Get indexes for the dataset """
    return [x for x in range(len(dataset))]


def sample(dataset, num_samples, replace=True):
    """ Sample the dataset """
    data_idx = dummy_indices(dataset)
    sample_idx = resample(data_idx, n_samples=num_samples, replace=replace)
    return Subset(dataset, sample_idx)


def leave_one_out_bootstrap(dataset, num_bootstraps, prop_train=0.5, seed=13):
    """ Create bootstrap samples 

    Parameters
    ----------
    dataset : datastore.dataset 

    num_bootstraps : int
        Number of bootstrap samples

    prop_train : float
        Proportion of samples to make up the training set
    
    seed : int
        Random seed to control the sampling

    Returns
    -------
    samples : list(namedtuple<Subset, Subset>)
        bootstrap samples of the data
    """
    BootstrapSample = namedtuple('BootstrapSample', 'train test')
    # Lean towards a slightly larger training set
    n_samples = math.ceil(len(dataset) * prop_train)
    data_idx = dummy_indices(dataset) 

    samples = []
    for i in range(num_bootstraps):
        train_idx = resample(data_idx, n_samples=n_samples)
        # Get dataset indices not included in training sample 
        test_idx = np.setdiff1d(data_idx, train_idx)

        sample = BootstrapSample(
            train = Subset(dataset, train_idx),
            test = Subset(dataset, test_idx)
        )

        samples.append(sample)

    return samples 