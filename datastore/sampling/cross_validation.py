from collections import namedtuple
from sklearn.model_selection import StratifiedKFold

from datastore.api.data import Subset


def stratified_split(dataset, num_splits, seed=42):
    """ Create stratified k-fold splits

    Parameters
    ----------
    dataset : datastore.dataset

    num_splits : int
        Number of splits of the data (usually denoted by `k` folds)

    seed : int
        Random seed to control the splits

    Returns
    -------
    splits : list(namedtuple<Subset, Subset>)
        stratified splits of the data
    """
    skf = StratifiedKFold(n_splits=num_splits, random_state=seed)
    data, labels = dataset.load_data()

    splits = []
    Split = namedtuple('Split', 'train valid')

    for train_idx, valid_idx in skf.split(data, labels):
        split = Split(
            train = Subset(dataset, train_idx),
            valid = Subset(dataset, valid_idx)
        )

        splits.append(split)

    return splits


def multitask_stratified_split(dataset, num_splits, label, seed=42):
    """ Create stratified k-fold splits

    Parameters
    ----------
    dataset : datastore.dataset

    num_splits : int
        Number of splits of the data (usually denoted by `k` folds)

    label : str
        Key to the labels dictionary. Determines which label to
        stratify on.

    seed : int
        Random seed to control the splits

    Returns
    -------
    splits : list(namedtuple<Subset, Subset>)
        stratified splits of the data
    """
    skf = StratifiedKFold(n_splits=num_splits, random_state=seed)
    data, labels = dataset.load_data()
    # Stratifiy using given `label`
    labels = labels[label]

    splits = []
    Split = namedtuple('Split', 'train valid')

    for train_idx, valid_idx in skf.split(data, labels):
        split = Split(
            train = Subset(dataset, train_idx),
            valid = Subset(dataset, valid_idx)
        )

        splits.append(split)

    return splits
