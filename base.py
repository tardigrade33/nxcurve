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



def draw_projection(X,X_r,color):
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
    return (X,color)


###### DATA ####################
## important variables
n_comp = 2
n_nei =12
nsamples = 1000

#Swissroll
#X, color = datasets.make_swiss_roll(n_samples=nsamples)
#Scurve
X, color = datasets.make_s_curve(n_samples=nsamples)

########## DR ##################
####Sklearn method
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=n_nei, n_components=n_comp)
# print("Done. Reconstruction error: %g" % err)

####Kernel Method

print("performing kernel calculation")
#LLE = LocallyLinearEmbedding(n_neighbors=n_nei)
#print(type(LLE))
#K = LLE.K(X)
#X_r = kernel_pca(X,K,n_comp)
#draw_projection(X,X_r,color)
#hdpd =    euclidean_distances(X,X)
#print(hdpd)
#ldpd =  euclidean_distances(X_r,X_r)
#print((np.array(X_r_dist)).shape)


# X = np.array([[ 2.22942149  ,7.85955028 ,13.79802659],
#     [10.9935585  ,10.69743902 ,-5.12635759],
#     [ 3.21392031  ,1.58486527  ,6.67122432],
#     [12.55779643  ,9.59765318  ,2.09874601],
#     [ 9.944827   ,10.27754689  ,8.81891858],])



# X_r =  np.array( [[-0.43301874  ,0.16590887],
#         [ 0.71433688  ,0.36647625],
#         [-0.43666503  ,0.47939415],
#         [ 0.30069242 ,-0.28622852],
#         [-0.14534552 ,-0.72555075],])

# hdpd =   euclidean_distances(X,X)
# #print(hdpd)
# ldpd =  euclidean_distances(X_r,X_r)
#print(ldpd)
# c = np.array([[5. ,0. ,0. ,0. ,0.],
#     [0. ,5. ,0. ,0. ,0.],
#     [0. ,0. ,2. ,1. ,2.],
#     [0. ,0. ,3. ,2. ,0.],
#     [0. ,0. ,0. ,2. ,3.],])
# a = rnx.coranking(hdpd,ldpd)

# n,x,p,b = rnx.nx_trusion(a)

# #print(rnx.difranking(X,X_r))
k=0
pts=10
rnx.nx_scores(k,pts,X,X)