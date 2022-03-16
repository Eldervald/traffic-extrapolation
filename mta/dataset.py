from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# Torch dataset. Used primary for batch training.
class DayObservationsDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.X = df['STATION'].to_numpy()
        self.targets = df['ridership'].to_numpy()
        self.observed_nodes = set(np.unique(self.X))

        self.node_to_target = df.set_index('STATION')['ridership'].to_dict()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]

    def get_observation_by_node(self, node):
        return self.node_to_target[node]

    def get_observed_nodes(self):
        return self.observed_nodes
    
    @classmethod
    def from_dataframe_by_day(cls, df, date):
        df = df[df['DATE'] == date].drop('DATE', axis=1)
        return cls(df)
