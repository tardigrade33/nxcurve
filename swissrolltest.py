import matplotlib.pyplot as plt
#from LLE import LocallyLinearEmbedding      #import LLE
# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D
#----------------------------------------------------------------------
# Locally linear embedding of the swiss roll
from sklearn import manifold, datasets
X, color = datasets.make_swiss_roll(n_samples=1500)
#print(X)
print("Computing LLE embedding")
#----------------------------------------------------------------------
#X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2) #sklearn method
#print("Done. Reconstruction error: %g" % err)
#print(X_r)
# Martin implementation

#print("kernel")
#LLE = LocallyLinearEmbedding()
#print(type(LLE))
#X_r = LLE.K(X)
#print(X_r)
#----------------------------------------------------------------------
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA
#X, _ = load_digits(return_X_y=True)
transformer = KernelPCA(n_components=3, kernel='linear')
X_transformed = transformer.fit_transform(X)
print(X_transformed.shape)
#----------------------------------------------------------------------
#LLE = LocallyLinearEmbedding.K(X)
#print("Done. Reconstruction error: %g" % err)

#----------------------------------------------------------------------
# Plot result

# fig = plt.figure()
#
# ax = fig.add_subplot(211, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
#
# ax.set_title("Original data")
# ax = fig.add_subplot(212)
# ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.axis('tight')
# plt.xticks([]), plt.yticks([])
# plt.title('Projected data')
# plt.show()