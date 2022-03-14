from typing import Tuple, Union, List
import numpy as np
from tqdm.notebook import tqdm as tqdm
from sklearn.neighbors import NearestNeighbors

# import node2vec

import torch
import torch.nn as nn

import torch_geometric as pyg


class BaseEstimator(nn.Module):
    def __init__(self,  pyg_graph: pyg.data.Data, observations: Tuple[List, List]) -> None:
        super().__init__()
        self.g = pyg_graph

        # dicts for fast indexing
        self.node_to_gidx = np.vectorize(dict(zip(self.g.id.detach().cpu().numpy(), range(len(self.g.id)))).get)

        self.change_observations(observations)

    def change_observations(self, observations):
        if observations is not None:
            self.obs_nodes = observations[0]
            self.obs_targets = observations[1]
        else:
            raise ValueError("empty observations")


class KnnEstimator(BaseEstimator):
    def __init__(self, pyg_graph: pyg.data.Data, observations: Tuple[List, List], neighbors_num=50) -> None:
        super().__init__(pyg_graph, observations)

        self.NEIGHBORS_NUM = neighbors_num

        self.build_knn()
    
    def build_knn(self):
        self.neighbors = NearestNeighbors(n_neighbors=self.NEIGHBORS_NUM, metric='haversine')
        obs_nodes_indices = self.node_to_gidx(self.obs_nodes)
        self.neighbors.fit(torch.vstack([self.g.lat[obs_nodes_indices], self.g.lng[obs_nodes_indices]]).T.detach().cpu())

    def get_kneighbors(self, X):
        dists, indices = self.neighbors.kneighbors(torch.vstack([self.g.lat[X], self.g.lng[X]]).T.detach().cpu())
        # converting dists to meters
        dists = dists * 6371 * 1000

        # skipping loc by itself
        if self.training:
            dists, indices = dists[:, 1:], indices[:, 1:]

        return torch.as_tensor(dists), torch.as_tensor(indices)