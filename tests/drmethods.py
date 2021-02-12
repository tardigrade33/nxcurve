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


import matplotlib.pyplot as plt
# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D


def sphere(n_samples):
    """
    Return matrix with datapoins of a sphere and colors
    """
    #Sphere Begin
    # Create our sphere.
    from sklearn.utils import check_random_state
    n_samples = 1000
    random_state = check_random_state(0)
    p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
    t = random_state.rand(n_samples) * np.pi

    # Sever the poles from the sphere.
    indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
    color = p[indices]
    x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
        np.sin(t[indices]) * np.sin(p[indices]), \
        np.cos(t[indices])

    X = np.array([x, y, z]).T
    return (X,color)

def draw_projection(X,X_r,color):
    """
    input: Original data, DR data, color
    Draws original figure and projection of dimentionality reduction
    """
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.get_cmap("Spectral")) 
    ax.set_title("Original data")
    ax = fig.add_subplot(212)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.get_cmap("Spectral"))
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.title('Projected data')
    plt.show()