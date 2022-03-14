import numpy as np
import node2vec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from math import radians, cos, sin, asin, sqrt


@np.vectorize
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    # lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


radians = np.vectorize(radians)


def load_embeddings(file_path):
    return node2vec.node2vec.gensim.models.KeyedVectors.load(file_path)


# train-val-test split
def make_data_loaders_from_dataset(ds, train_batch_size=64):
    indices = list(range(len(ds)))

    train_indices, test_indices = train_test_split(indices, test_size=0.15)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.15)
    # print(len(train_indices), len(val_indices), len(test_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(ds, batch_size=train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(ds, batch_size=len(val_sampler.indices), sampler=val_sampler)
    test_loader = DataLoader(ds, batch_size=len(test_sampler.indices), sampler=test_sampler)

    return train_loader, val_loader, test_loader
