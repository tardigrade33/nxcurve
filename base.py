import matplotlib.pyplot as plt
import numpy as np
from Kernels.LLE import LocallyLinearEmbedding      #import LLE
# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from sklearn import manifold, datasets  #Datasets
from scipy.linalg import eigh
#pairwise distances
from sklearn.metrics.pairwise import euclidean_distances




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
nsamples = 10

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
X_dist =    euclidean_distances(X,X)
#print(X_dist)
X_r_dist =  euclidean_distances(X_r,X_r)
#print((np.array(X_r_dist)).shape)


def ranks(X_dist,nsamples):
    """
    input
        X_dist:   distances matrix
        nsamples; samples
    output
        ρij = |{k : δik < δij or (δik = δij and 1 ≤ k < j ≤ N )}|
    review that  ρij != ρik for k != j, even if δij = δik .
    """
    rows = len(X_dist)
    cols = len(X_dist[0])
    rank = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            for k in range(rows):
                count = []
                if X_dist[i][k] < X_dist[i][j] or ((X_dist[i][k] == X_dist[i][j]) and (1<=k and k<j and j<=nsamples)):
                    count.append(k)
            print(count)
            rank[i][j] = len(count)
    return rank

def ranking(X_dist,nsamples):
    """
    input
        X_dist:   distances matrix
        nsamples; samples
    output
        ρij = |{k : δik < δij or (δik = δij and 1 ≤ k < j ≤ N )}|
    review that  ρij != ρik for k != j, even if δij = δik .
    """
    idx = np.argsort(X_dist, axis=0)
    rows = len(X_dist)
    cols = len(X_dist[0])
    print(rows,cols)
    rank = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            rank[(idx[i][j])][j] = i
    print(rank)
    corank = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            h=int(rank[(idx[i][j])][j])
            #print(h)
            corank[i][h] =  corank[i][h] + 1
    return corank


print(ranking(X_r_dist,nsamples))
