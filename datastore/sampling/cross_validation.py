from collections import namedtuple
from sklearn.model_selection import StratifiedKFold
from datastore.api.data import Subset


def stratified_split(dataset, num_splits):
    """ Create stratified k-fold splits

    Parameters
    ----------
    dataset : datastore.dataset

    num_splits : int

    Returns
    -------
    splits : list(namedtuples<subset, subset>)
        stratified splits of the data
    """
    skf = StratifiedKFold(n_splits=num_splits)
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
