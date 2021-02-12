
import numpy as np
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.neighbors import NearestNeighbors


def barycenter_weights(X, Z, reg=1e-3):
    X = check_array(X, dtype=FLOAT_DTYPES)
    Z = check_array(Z, dtype=FLOAT_DTYPES, allow_nd=True)

    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B


def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    
    knn = NearestNeighbors(n_neighbors=(n_neighbors + 1), n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))




def locally_linear_embedding(
        X, n_neighbors,  reg=1e-3, eigen_solver='auto', 
        max_iter=100,random_state=None, n_jobs=None):

    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

   
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" %
            (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = (eigen_solver != 'dense')

    W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)

        # we'll compute M = (I-W)'(I-W)
        # depending on the solver, we'll do this differently
    if M_sparse:
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

    return M

class LocallyLinearEmbedding:
    def __init__(self, n_neighbors=6, reg=1E-3,
                 eigen_solver='auto', max_iter=100,
                 neighbors_algorithm='auto', random_state=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def K(self, X):
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      n_jobs=self.n_jobs)

        random_state = check_random_state(self.random_state)
        X = check_array(X, dtype=float)
        self.nbrs_.fit(X)
        n=len(X)
        M= locally_linear_embedding(
                self.nbrs_, self.n_neighbors,
                eigen_solver=self.eigen_solver, 
                max_iter=self.max_iter, 
                random_state=random_state, reg=self.reg, n_jobs=self.n_jobs)
        lambdamax, evec = largest_eigsh(M, 1, which='LM')
        L=lambdamax*np.eye(n)-M
        """ Center L
        m=len(L)
        e=1/np.sqrt(m)*np.ones([m,1])
        S=np.eye(len(L))-np.matmul(e.T,e)
        L=np.matmul(S,L,S)
        """
        return L

##############################HOW TO USE THE KERNEL######################
#LLE=LocallyLinearEmbedding() 
#print(type(LLE))

#LLE.K()
