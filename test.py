#rnxfun.py
# Author: Nicolas Marin <josue.marin1729@gmail.com>
# License: MIT
from sklearn import manifold, datasets  #Datasets
import numpy as np
import qualitycurve as qc
import drmethods as drm

###### DATA ####################
## important variables
n_comp = 2
n_nei = 20
nsamples = 1000

#Swissroll
X = np.loadtxt('/home/tardigrade/Documents/yachay/thesis/AWESOME/CODE/swiss.dat', dtype=float)
#X, color = datasets.make_swiss_roll(n_samples=nsamples)
#X, color = drm.sphere(n_samples=nsamples)
#Scurve
#X, color = datasets.make_s_curve(n_samples=nsamples)

########## DR ##################
####Sklearn method
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=n_nei, n_components=n_comp)

####Kernel Method use wrapers KPCA
#X_r = drm.lle(X,n_comp,n_nei)
#drmeth.draw_projection(X,X_r,color)

#drm.draw_projection(X,X_r,color)

qc.quality_curve(X,X_r)