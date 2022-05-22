from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# Torch dataset. Used primary for batch training.
class DayObservationsDataset(Dataset):
    def __init__(self, observations_df: pd.DataFrame) -> None:
        super().__init__()
        self.data = [str(x) for x in observations_df['id'].values]
        self.targets = observations_df['pickups'].to_numpy()
        self.observed_nodes = set(np.unique(self.data))

        self.node_to_target = observations_df.set_index('id')['pickups'].to_dict()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_observation_by_node(self, node):
        return self.node_to_target[node]

    def get_observed_nodes(self):
        return self.observed_nodes
    
    @classmethod
    def from_dataframe_by_day(cls, observations_df, day):
        df = observations_df[observations_df['day'].astype('int') == day].copy()
        df.drop('day', axis=1, inplace=True)
        return cls(df)
