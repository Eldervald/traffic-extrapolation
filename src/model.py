from typing import Tuple, List
import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn

import torch_geometric as pyg


class BaseEstimator(nn.Module):
    def __init__(self,  pyg_graph: pyg.data.Data, nodes, targets) -> None:
        super().__init__()
        self.g = pyg_graph

        # dicts for fast indexing
        self.node_to_idx = np.vectorize(dict(zip(self.g.id, range(len(self.g.id)))).get)

        self._set_observations(nodes, targets)
    
    def set_observations(self, nodes, targets):
        return self._set_observations(nodes, targets)

    def _set_observations(self, nodes, targets):
        if nodes is not None and targets is not None:
            self.obs_nodes = np.asarray(nodes)
            self.obs_targets = np.asarray(targets)
            self.obs_node_indices = np.asarray([int(self.node_to_idx(x)) for x in self.obs_nodes])
        else:
            raise ValueError("empty observations")


class KnnEstimator(BaseEstimator):
    def __init__(self, pyg_graph: pyg.data.Data, nodes, targets, neighbors_num=10) -> None:
        super().__init__(pyg_graph, nodes, targets)

        self.neighbors_num = neighbors_num

        self.build_knn()
    
    def set_observations(self, nodes, targets):
        super().set_observations(nodes, targets)
        self.build_knn()
    
    def build_knn(self):
        self.neighbors = NearestNeighbors(n_neighbors=self.neighbors_num, metric='haversine')
        self.neighbors.fit(torch.vstack([self.g.lat[self.obs_nodes_indices],
                                         self.g.lon[self.obs_nodes_indices]]).T.detach().cpu())

    def get_kneighbors_with_observations(self, X):
        """Returns dists, neighbors indices and corresponding targets

        Args:
            X (np.array): nodes indices

        Returns:
            (torch.tensor, torch.tensor, torch.tensor): dists, neighbors indices and corresponding targets
        """
        n_neighbors = self.neighbors_num + (self.training == True)

        dists, indices = self.neighbors.kneighbors(torch.vstack([self.g.lat[X], self.g.lon[X]]).T.detach().cpu(), 
            n_neighbors=n_neighbors)

        # converting dists to km
        dists = dists * 6371

        # skipping loc by itself
        if self.training:
            dists, indices = dists[:, 1:], indices[:, 1:]

        dists = torch.as_tensor(dists, dtype=torch.float32)
        neighbors_indices = torch.as_tensor(self.node_to_idx(self.obs_nodes[indices]))
        targets = torch.as_tensor(self.obs_targets[indices], dtype=torch.float32)

        return dists, neighbors_indices, targets