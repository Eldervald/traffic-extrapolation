import numpy as np


def propEmbed(A, alpha = 0.95):
    #A is the adjacency matrix, alpha is a hyperparameter (random walk probability)
    #embedding through the stable state of the random walker probability distribution
    #random walk model is similar to pagerank but wrt given home node - with probability alpha randomly following the edges with the probability proportional to the edge weights
    #and (1-alpha) chance of a teleport, but not to a random node but to a home node
    #then the final stable state probability distribution defined below depends on the home node
    # and this way provides an n-dim embedding of each node
    A = A / A.sum(axis = 1)
    n = A.shape[0]
    AI = np.linalg.inv(np.eye(n) - A * alpha)
    X = (1 - alpha) * AI
    return X