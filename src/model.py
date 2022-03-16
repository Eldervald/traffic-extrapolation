from typing import Tuple, List
import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn

import torch_geometric as pyg


class BaseEstimator(nn.Module):
    def __init__(self,  pyg_graph: pyg.data.Data, obs_nodes, obs_targets) -> None:
        super().__init__()
        self.g = pyg_graph

        # dicts for fast indexing
        self.node_to_idx = np.vectorize(dict(zip(self.g.id, range(len(self.g.id)))).get)

        self._set_observations(obs_nodes, obs_targets)
    
    def set_observations(self, obs_nodes, obs_targets):
        return self._set_observations(obs_nodes, obs_targets)

    def _set_observations(self, obs_nodes, obs_targets):
        if obs_nodes is not None and obs_targets is not None:
            self.obs_nodes = np.asarray(obs_nodes)
            self.obs_targets = obs_targets
        else:
            raise ValueError("empty observations")


class KnnEstimator(BaseEstimator):
    def __init__(self, pyg_graph: pyg.data.Data, obs_nodes, obs_targets, neighbors_num=10) -> None:
        super().__init__(pyg_graph, obs_nodes, obs_targets)

        self.neighbors_num = neighbors_num

        self.build_knn()
    
    def set_observations(self, obs_nodes, obs_targets):
        super().set_observations(obs_nodes, obs_targets)
        self.build_knn()
    
    def build_knn(self):
        self.neighbors = NearestNeighbors(n_neighbors=self.neighbors_num, metric='haversine')
        obs_nodes_indices = np.asarray([int(self.node_to_idx(x)) for x in self.obs_nodes])
        self.neighbors.fit(torch.vstack([self.g.lat[obs_nodes_indices], self.g.lon[obs_nodes_indices]]).T.detach().cpu())

    def get_kneighbors(self, X):
        dists, indices = self.neighbors.kneighbors(torch.vstack([self.g.lat[X], self.g.lon[X]]).T.detach().cpu())
        # converting dists to meters
        dists = dists * 6371 * 1000

        # skipping loc by itself
        if self.training:
            dists, indices = dists[:, 1:], indices[:, 1:]

        return torch.as_tensor(dists), torch.as_tensor(indices)