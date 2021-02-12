"""Spectral Embedding"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause
import numpy as np

import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import lobpcg
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

def spectral_embedding(adjacency, norm_laplacian=True):
    adjacency = check_symmetric(adjacency)

    
    n_nodes = adjacency.shape[0]

    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                         return_diag=True)
    return np.linalg.pinv(laplacian.todense())
    


class SpectralEmbedding(BaseEstimator):

    def __init__(self, n_neighbors=2, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    
    def _get_affinity_matrix(self, X):
        self.n_neighbors_ = (self.n_neighbors
                                     if self.n_neighbors is not None
                                     else max(int(X.shape[0] / 10), 1))
        self.affinity_matrix_ = kneighbors_graph(X, self.n_neighbors_,
                                                         include_self=True,
                                                         n_jobs=self.n_jobs)
                # currently only symmetric affinity_matrix supported
        self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ +
                                               self.affinity_matrix_.T)
        return self.affinity_matrix_

    def K(self, X):
        """Fit the model from data in X.
        """

        X = check_array(X, accept_sparse='csr', ensure_min_samples=2,
                        estimator=self)

        
        affinity_matrix = self._get_affinity_matrix(X)
        return spectral_embedding(affinity_matrix)
        
LE=SpectralEmbedding()






