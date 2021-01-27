import matplotlib.pyplot as plt
from Kernels.LLE import LocallyLinearEmbedding      #import LLE
# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from sklearn import manifold, datasets  #Datasets
from scipy.linalg import eigh
#pairwise distances
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import rnxfun as rnx



def draw_projection(X,X_r):
    """
    Draws original figure and Reduction
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

def sphere(n_samples):
    """
    Return matrix with datapoins of a sphere
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
    return X,color


###### DATA ####################
## important variables
n_comp = 2
n_nei =2
nsamples = 5

#Swissroll
X, color = datasets.make_swiss_roll(n_samples=nsamples)
#Scurve
#X, color = datasets.make_s_curve(n_samples=nsamples)

########## DR ##################
####Sklearn method
# X_r, err = manifold.locally_linear_embedding(X, n_neighbors=n_nei, n_components=n_comp)
# print("Done. Reconstruction error: %g" % err)

####Kernel Method

print("performing kernel calculation")
LLE = LocallyLinearEmbedding(n_neighbors=n_nei)
print(type(LLE))
K = LLE.K(X)
X_r = kernel_pca(X,K,n_comp)
#draw_projection(X,X_r)
hdpd =    euclidean_distances(X,X)
#print(hdpd)
ldpd =  euclidean_distances(X_r,X_r)
#print((np.array(X_r_dist)).shape)

c = np.array([[5. ,0. ,0. ,0. ,0.],
    [0. ,5. ,0. ,0. ,0.],
    [0. ,0. ,2. ,1. ,2.],
    [0. ,0. ,3. ,2. ,0.],
    [0. ,0. ,0. ,2. ,3.],])
a = rnx.coranking(hdpd,ldpd)
print(rnx.nx_trusion(a)[1])