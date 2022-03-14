from torch.utils.data import Dataset
import pandas as pd
import numpy as np


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
