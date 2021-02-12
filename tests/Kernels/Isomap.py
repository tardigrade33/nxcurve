
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier,kneighbors_graph
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.graph import graph_shortest_path
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

import numpy as np

class Isomap:
    def __init__(self, n_neighbors=2,path_method='auto',
                neighbors_algorithm='auto', n_jobs=None, metric='minkowski',
                p=2, metric_params=None):
       self.n_neighbors = n_neighbors
       self.path_method = path_method
       self.neighbors_algorithm = neighbors_algorithm
       self.n_jobs = n_jobs
       self.metric = metric
       self.p = p
       self.metric_params = metric_params
    def GeoDesicMatrix(self, X):

       self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                     algorithm=self.neighbors_algorithm,
                                     metric=self.metric, p=self.p,
                                     metric_params=self.metric_params,
                                     n_jobs=self.n_jobs)
       self.nbrs_.fit(X)

       kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                              metric=self.metric, p=self.p,
                              metric_params=self.metric_params,
                              mode='distance', n_jobs=self.n_jobs)

       self.dist_matrix_ = graph_shortest_path(kng,
                                               method=self.path_method,
                                               directed=False)
       G = self.dist_matrix_ ** 2
       return G
    def H(self,X):
        self.n=len(X)
        e=np.ones([self.n,1])
        eT=np.transpose(e)
        return np.identity(self.n)-1/self.n*np.matmul(e,eT)

    def K_(self,X):
        H=self.H(X)
        D=self.GeoDesicMatrix(X)
        K=-0.5*np.matmul(np.matmul(H,D),H)
        return K
    def K(self,X):
        n=len(X)
        H=self.H(X)
        D=self.GeoDesicMatrix(X)
        K=-0.5*np.matmul(np.matmul(H,D),H)
        M=np.block([[ np.zeros([n,n]),2*K],[ np.eye(n),-4*K]])
        evals_large_sparse, evec = largest_eigsh(M, 1, which='LM')
        c=evals_large_sparse[0]
        return K+2*c*K+0.5*c*c*H

##############################HOW TO USE THE KERNEL######################

Iso=Isomap() #CALL CLASS
