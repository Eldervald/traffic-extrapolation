import numpy as np
import node2vec
from node2vec import Node2Vec
import networkx as nx

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


def make_graph_from_df(nodes_df, edges_df, name='TLC', directed=False):
    G = nx.Graph(directed=directed)
    G.graph['Name'] = name

    G.add_nodes_from(nodes_df.set_index('id').to_dict('index').items())
    G.add_nodes_from((n, {'id': n}) for n in G.nodes())

    G.add_edges_from(nx.from_pandas_edgelist(edges_df, 'oid', 'did', ['dist']).edges(data=True))

    return G


def generate_node_embeddings(G):
    model = Node2Vec(G, dimensions=128, walk_length=30, workers=4)
    embeddings = model.fit(window=5, min_count=1)
    return embeddings


def load_embeddings(file_path):
    return node2vec.node2vec.gensim.models.KeyedVectors.load(file_path)
