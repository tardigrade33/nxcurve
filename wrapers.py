#wrapers
from Kernels.Isomap import Isomap
from Kernels.LLE import LocallyLinearEmbedding
from Kernels.laplacian import SpectralEmbedding
import numpy as np
from scipy.linalg import eigh

def kernel_pca(X,K,n_components):
    """
    Input:  matrix de datos, matrix kernel, numero de componentes
    Output: PCA
    """

    print("performing dimentionality reduction")
    #Dimention Reduction
    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)
    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    return X_pc


def isomap(X,n_comp,n_nei):
    """
    input:  data X, number of components, n neighbors
    output: dimentionality reduction usin isomap
    """
    ISO = Isomap(n_neighbors=n_nei)
    K= ISO.K(X)
    X_r = kernel_pca(X,K,n_comp)
    return X_r

def lle(X,n_comp,n_nei):
    """
    input:  data X, number of components, n neighbors
    output: dimentionality reduction usin lle
    """
    LLE = LocallyLinearEmbedding(n_neighbors=n_nei)
    K = LLE.K(X)
    X_r = kernel_pca(X,K,n_comp)
    return X_r
def le(X,n_comp,n_nei):
    """
    input:  data X, number of components, n neighbors
    output: dimentionality reduction usin laplacian eigenmaps
    """
    SP = SpectralEmbedding(n_neighbors=n_nei)
    K = SP.K(X)
    X_r = kernel_pca(X,K,n_comp)
    return X_r

