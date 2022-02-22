from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


RANDOM_STATE = 123


# Torch dataset. Used primary for batch training.
class DayObservationsDataset(Dataset):
    def __init__(self, pickups_df: pd.DataFrame) -> None:
        super().__init__()
        self.data = pickups_df['id'].to_numpy()
        self.targets = pickups_df['pickups'].to_numpy()
        self.observed_nodes = set(np.unique(self.data))

        self.node_to_target = pickups_df.set_index('id')['pickups'].to_dict()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_observation_by_node(self, node):
        return self.node_to_target[node]

    def get_observed_nodes(self):
        return self.observed_nodes
    
    @classmethod
    def from_dataframe_by_day(cls, pickups_df, day):
        df = pickups_df[pickups_df['day'].astype('int') == day].copy()
        df.drop('day', axis=1, inplace=True)
        return cls(df)


# train-val-test split
def make_data_loaders_from_dataset(ds, train_batch_size=64):
    indices = list(range(len(ds)))

    train_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=RANDOM_STATE)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=RANDOM_STATE)
    # print(len(train_indices), len(val_indices), len(test_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(ds, batch_size=train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(ds, batch_size=len(val_sampler.indices), sampler=val_sampler)
    test_loader = DataLoader(ds, batch_size=len(test_sampler.indices), sampler=test_sampler)

    return train_loader, val_loader, test_loader