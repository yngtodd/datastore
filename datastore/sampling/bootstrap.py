import math
import numpy as np

from collections import namedtuple
from sklearn.utils import resample

from datastore.api.data import Subset


def bootstrap(dataset, num_bootstraps, prop_train=0.5, seed=13):
    """ Create bootstrap samples """

    BootstrapSample = namedtuple('BootstrapSample', 'train valid')
    n_samples = math.ceil(len(dataset) * prop_train)
    data_idx = [x for x in range(len(dataset))]

    samples = []
    for i in range(num_bootstraps):
        train_idx = resample(data_idx, n_samples=n_samples)
        valid_idx = np.setdiff1d(data_idx, train_idx)

        sample = BootstrapSample(
            train = Subset(dataset, train_idx),
            valid = Subset(dataset, valid_idx)
        )

        samples.append(sample)

    return samples 